"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

code base on: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 1000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
DECAY = 0.99    # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)   # (s a r s_)
        self.pointer = 0
        self.sigm = 3
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,     # dimmensions of action, state
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # actor and critic
        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        ema = tf.train.ExponentialMovingAverage(decay = DECAY)     # Exponential Moving Average

        # apply(var_list) creates shadow variables for all elements of var_list, and returns an Operation that updates the moving averages.
        self.target_update = [ema.apply(a_params), ema.apply(c_params)]

        # overwrite the default custom_getter function (ema.average() to the default getter).
        # a_params[0]: <tf.Variable 'Actor/l1/kernel:0' shape=(3, 30) dtype=float32_ref>
        # ema.average(a_params[0]): <tf.Variable 'Actor/l1/kernel/ExponentialMovingAverage:0' shape=(3, 30) dtype=float32_ref>
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        # target actor and critic
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)    # ema.averager(var): Returns the Variable holding the average of var.
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        # train actor
        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        # train critic
        q_target = self.R + GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        print("\n--------------------------\n")
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # update actor and critic network
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        # update target network
        self.sess.run(self.target_update)

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_,done))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    # Actor network
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = not reuse
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')   # tf.multiply returns x*y element-wise.
    # Critic network
    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = not reuse
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = 3  # control exploration
    t1 = time.time()
    low_states = []
    for i in range(MAX_EPISODES):
        s = env.reset()
        s0 = s
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward <= -900:
                    print('Episode:',i, s0)
                    low_states.append((ep_reward,s0))
                # if ep_reward > -300:RENDER = True
                break
    #
    #     if ep_reward > -10 and i > 100:
    #         break
    #
    #
    # while True:
    #     s = env.reset()
    #     for j in range(MAX_EP_STEPS):
    #         env.render()
    #         a = ddpg.choose_action(s)
    #         s, r, done, info = env.step(a)



    print('Running time: ', time.time() - t1)
    for _ in low_states:
        print(_)