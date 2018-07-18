import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
import gym
from gym.spaces import Box, Discrete, Tuple
from modules import policies
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

HIDDEN = 15


class LstmPolicy(object):

    def __init__(self, X, nbatch, nsteps, nlstm=50, reuse=False):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(Discrete(13))
        # X, processed_x = observation_input(ob_space, nbatch)

        # X = tf.placeholder(tf.float32, [nbatch, HIDDEN])
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("ppo", reuse=reuse):
            h = X
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        a_z = tf.placeholder(tf.float32, [nbatch, 1])

        print(a0)
        print(a_z)
        neglogp0 = self.pd.neglogp(a0)
        neglogpz = self.pd.neglogp(a_z)

        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        # self.X = X
        self.M = M
        self.S = S
        self.vf = vf


        self.a0 = a0
        self.a_z = a_z
        self.v0 = v0
        self.snew = snew
        self.neglogp0 = neglogp0
        self.neglogpz = neglogpz
        


class PPO(object):
    def __init__(self, args, z, ent_coef, vf_coef, max_grad_norm):
        # sess = tf.get_default_session()

        self.args = args
        act_model = LstmPolicy(z, args['batch_size'], 1, reuse=False)
        train_model = LstmPolicy(z, args['batch_size'], args['batch_size'], reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        CLIPRANGE = 0.2

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        tf.summary.scalar('pgloss', pg_loss)
        tf.summary.scalar('vfloss', vf_loss)
        tf.summary.scalar('entropyloss', entropy)
        # with tf.variable_scope('model'):
        #     params = tf.trainable_variables()
        # grads = tf.gradients(loss, params)
        # if max_grad_norm is not None:
        #     grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # grads = list(zip(grads, params))
        # trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # _train = trainer.apply_gradients(grads)

        # self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        # def save(save_path):
        #     ps = sess.run(params)
        #     joblib.dump(ps, save_path)

        # def load(load_path):
        #     loaded_params = joblib.load(load_path)
        #     restores = []
        #     for p, loaded_p in zip(params, loaded_params):
        #         restores.append(p.assign(loaded_p))
        #     sess.run(restores)
        #     # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.loss = loss
        self.train_model = train_model
        self.act_model = act_model
        self.initial_state = act_model.initial_state


	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args['ppo']['learning_rate'])
		gvs = self.opt.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ppo'))
		capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
		opt_op = self.opt.apply_gradients(capped_gvs)
		return opt_op

	def variable_restore(self, sess):

		model_filename = os.path.join("save", 'ppo')

		if os.path.isfile(model_filename + '.meta'):
			self.saver = tf.train.import_meta_graph(model_filename + '.meta')
			self.saver.restore(sess, model_filename)
			return

		if os.path.isfile(model_filename):
			self.saver.restore(sess, model_filename)
			return