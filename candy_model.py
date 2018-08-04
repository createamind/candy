# -*- coding: utf-8 -*-

from modules.c3d import C3D_Encoder

from modules.policy_gradient import PG, PGLoss

from modules.deconv import ImageDecoder

from modules.losses import MSELoss, CrossEntropyLoss

from modules.networks import MLP

from modules.vae import VAE, VAELoss

from modules.ppo import PPO, LstmPolicy

import tensorflow as tf
import numpy as np
import yaml
import datetime
import functools

print = functools.partial(print, flush=True)

class ARGS(object):
	pass

class Machine(object):
	def __init__(self):

		args = self.get_args()
		self.args = args

		#Building Graph
		self.raw_image = tf.placeholder(tf.float32, shape=(args['batch_size'], 320, 320, 8))
		self.speed = tf.placeholder(tf.float32, shape=(args['batch_size'], 1))

		self.test_raw_image = tf.placeholder(tf.float32, shape=(1, 320, 320, 8))
		self.test_speed = tf.placeholder(tf.float32, shape=(1, 1))

		#[self.image_sequence, self.raw_image, self.depth_image, self.seg_image, self.speed, self.collision, self.intersection, self.control, self.reward, self.transition]

		# self.c3d_encoder = C3D_Encoder(args,'c3d_encoder', inputs[0])
		# self.c3d_future = C3D_Encoder(args,'c3d_encoder', inputs[9], reuse=True)

		self.vae = VAE(args, 'vae', self.raw_image, reuse=False)
		self.test_vae = VAE(args, 'vae', self.test_raw_image, reuse=True)

		# self.future_vae = VAE(args, self.c3d_future.inference())

		recon_x, z, logsigma = self.vae.inference()
		self.vae_loss = VAELoss(args, 'vae', recon_x, self.raw_image, z, logsigma)

		test_recon_x, test_z, test_logsigma = self.test_vae.inference()
		self.test_vae_loss = VAELoss(args, 'vae', test_recon_x, self.test_raw_image, test_z, test_logsigma)

		z = tf.concat([z, self.speed], 1)
		test_z = tf.concat([test_z, self.test_speed], 1)

		z = tf.clip_by_value(z, -5, 5)
		test_z = tf.clip_by_value(test_z, -5, 5)

		# z = tf.Print(z, [z[0]], summarize=15)
		# test_z = tf.Print(test_z, [test_z[0]], summarize=20)

		self.ppo = PPO(args, 'ppo', z=z, test_z=test_z, ent_coef=.01, vf_coef=0.5, max_grad_norm=0.5)

		self.test_vae_loss.inference()
		# z = self.c3d_encoder.inference()

		# self.raw_decoder = ImageDecoder(args, 'raw_image', z, last=3)
		# self.raw_decoder_loss = MSELoss(args, 'raw_image', self.raw_decoder.inference(), inputs[1])

		# self.seg_decoder = ImageDecoder(args, 'seg', z, last=13)
		# self.seg_decoder_loss = CrossEntropyLoss(args, 'seg', self.seg_decoder.inference(), inputs[3])

		# self.depth_decoder = ImageDecoder(args, 'depth', z, last=1)
		# self.depth_decoder_loss = MSELoss(args, 'depth', self.depth_decoder.inference(), inputs[2])

		# self.speed_prediction = MLP(args, 'speed', z, 1, 300)
		# self.speed_loss = MSELoss(args, 'speed', self.speed_prediction.inference(), inputs[4])        

		# self.collision_prediction = MLP(args, 'collision', z, 1, 300)
		# self.collision_loss = MSELoss(args, 'collision', self.collision_prediction.inference(), inputs[5])

		# self.intersection_prediction = MLP(args, 'intersection', z, 1, 300)
		# self.intersection_loss = MSELoss(args, 'intersection', self.intersection_prediction.inference(), inputs[6])

		# self.policy = PG(args, 'policy', z, 13)
		# self.log_probs = self.policy.inference()
		# self.policy_loss = PGLoss(args, 'policy', inputs[7], inputs[8], self.log_probs)

		# self.value = MLP(args, 'value', z, 1, 300)

		# self.transition = MLP(args, 'transition', tf.concat([z, self.log_probs],1), 300, 300)
		# self.transition_loss = MSELoss(args, 'transition', self.transition.inference(), self.c3d_future.inference())

		# self.imitation_loss = CrossEntropyLoss(args, self.policy.inference(), inputs[7])
		# self.reward_loss = MESLoss(args, self.value.inference(), inputs[8])


		# # MCTS
		# self.z_mcts = tf.placeholder(tf.float32, shape=(1, 100))
		# self.policy_mcts = MLP(args, 'policy', self.z_mcts, 36, 100).inference()
		# self.value_mcts = MLP(args, 'value', self.z_mcts, 1, 100).inference()
		# self.transition_mcts = MLP(args, 'transition', self.z_mcts, 100, 100).inference()

		# self.mcts = MCTS('mcts', self.policy_inference, self.value_inference, self.transition_inference)
		# self.action = self.mcts.inference()
		#Structures with variables    
		# self.intersection_lane = MLP('intersection_lane')
		# self.intersection_offroad = MLP('intersection_offroad') 

		# Process Steps
		# self.mcts = MCTS('mcts')

		# self.transition = TransitionNetwork('transition')
		# self.policy = PolicyNetwork('policy')
		# self.safety = ValueNetwork('safety')
		# self.goal = ValueNetwork('goal')

		# self.variable_parts = [self.c3d_encoder, self.raw_decoder, self.seg_decoder, self.depth_decoder]
		self.variable_parts = [self.vae, self.ppo, self.test_vae]
		self.variable_parts2 = [self.vae, self.ppo]
		# self.variable_parts2 = []
		# self.variable_parts = [self.c3d_encoder, self.raw_decoder]

		# self.variable_parts = [self.c3d_encoder, self.raw_decoder, self.seg_decoder, self.depth_decoder, \
		# 	self.speed_prediction, self.collision_prediction, self.intersection_prediction, self.policy]

		# self.loss_parts = self.collision_loss.inference() + self.intersection_loss.inference() + self.speed_loss.inference() + self.depth_decoder_loss.inference() + \
		# 			self.raw_decoder_loss.inference() + self.seg_decoder_loss.inference() + self.policy_loss.inference() + self.transition_loss.inference()

		# self.variable_parts = [self.c3d_encoder, self.raw_decoder, self.seg_decoder, self.depth_decoder, \
		# 	self.speed_prediction, self.collision_prediction, self.intersection_prediction, self.policy]

		# self.loss_parts = self.collision_loss.inference() + self.intersection_loss.inference() + self.speed_loss.inference() + self.depth_decoder_loss.inference() + \
		# 			self.raw_decoder_loss.inference() + self.seg_decoder_loss.inference() + self.policy_loss.inference()

		# self.loss_parts = self.depth_decoder_loss.inference() +self.raw_decoder_loss.inference() +self.seg_decoder_loss.inference()
		
		self.loss_parts = self.vae_loss.inference() + self.ppo.loss
		# self.loss_parts = self.raw_decoder_loss.inference()
				
		# weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses'))
		# tf.summary.scalar('weight_decay_loss', weight_decay_loss)
		total_loss = self.loss_parts
		tf.summary.scalar('total_loss', tf.reduce_mean(total_loss))

		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
	
		self.final_ops = []
		for part in self.variable_parts:
			self.final_ops.append(part.optimize(total_loss))
		self.final_ops = tf.group(self.final_ops)

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True


		self.merged = tf.summary.merge_all()
		self.sess = tf.Session(config = config)
		self.writer = tf.summary.FileWriter('logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), self.sess.graph)


		self.sess.run(tf.global_variables_initializer())

		print('Restoring!')

		for part in self.variable_parts:
			part.variable_restore(self.sess)

		print('Model Started!')

	def get_args(self):
		with open("args.yaml", 'r') as f:
			try:
				t = yaml.load(f)
				return t
			except yaml.YAMLError as exc:
				print(exc)
		

	# def ppotrain(self, inputs, z):
	# 	obs, actions, values, neglogpacs, rewards, _, states = inputs
	# 	model.train(self.args['learning_rate'], 0.2, obs, returns, actions, values, neglogpacs, states)

	#     # mblossvals = []

	# 	# assert nenvs % nminibatches == 0
	# 	# envsperbatch = nenvs // nminibatches
	# 	# envinds = np.arange(nenvs)
	# 	# flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
	# 	# envsperbatch = nbatch_train // nsteps
	# 	# for _ in range(noptepochs):
	# 	# 	np.random.shuffle(envinds)
	# 	# 	for start in range(0, nenvs, envsperbatch):
	# 	# 		end = start + envsperbatch
	# 	# 		mbenvinds = envinds[start:end]
	# 	# 		mbflatinds = flatinds[mbenvinds].ravel()
	# 	# 		slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
	# 	# 		mbstates = states[mbenvinds]
	# 	# 		mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

	def step(self, obs, state):
		# mask = np.zeros(1)
		td_map = {self.ppo.act_model.S:state}
		td_map[self.test_raw_image] = np.array([obs[0]])# frame输入
		td_map[self.test_speed] = np.array([[obs[1]]])# speed

		return self.sess.run([self.ppo.act_model.a0, self.ppo.act_model.v0, self.ppo.act_model.snew, self.ppo.act_model.neglogp0, self.test_vae_loss.recon], td_map)


	def value(self, obs, state, action):
		# mask = np.zeros(1)
		td_map = {self.ppo.act_model.S:state, self.ppo.act_model.a_z: [action]}
		td_map[self.test_raw_image] = np.array([obs[0]])
		td_map[self.test_speed] = np.array([[obs[1]]])
		return self.sess.run([self.ppo.act_model.a_z, self.ppo.act_model.v0, self.ppo.act_model.snew, self.ppo.act_model.neglogpz, self.test_vae_loss.recon], td_map)

	def train(self, inputs, global_step):
		obs, actions, values, neglogpacs, rewards, vaerecons, states, std_actions, manual = inputs

		# print(obs.shape)
		# print(actions.shape)
		# print(values.shape)
		# print(neglogpacs.shape)
		# print(rewards.shape)
		# print(vaerecons.shape)
		# print(states.shape)

		values = np.squeeze(values, 1)
		neglogpacs = np.squeeze(neglogpacs, 1)
		# rewards = np.squeeze(rewards, 1)

		raw_image = np.array([ob[0] for ob in obs])
		speed = np.array([[ob[1]] for ob in obs])

		# print(raw_image.shape)
		# print(speed.shape)

		advs = rewards - values
		advs = (advs - advs.mean()) / (advs.std() + 1e-5)

		td_map = {self.ppo.A:actions, self.ppo.ADV:advs, self.ppo.R:rewards, self.ppo.OLDNEGLOGPAC:neglogpacs, self.ppo.OLDVPRED:values}

		# mask = np.zeros(self.args['batch_size'])
		td_map[self.ppo.train_model.S] = np.squeeze(states, 1)
		# td_map[self.ppo.train_model.M] = mask

		td_map[self.ppo.std_action] = std_actions
		td_map[self.ppo.std_mask] = manual

		td_map[self.raw_image] = raw_image
		td_map[self.speed] = speed
		td_map[self.test_raw_image] = [raw_image[0]]
		td_map[self.test_speed] = [speed[0]]

		summary, _ = self.sess.run([self.merged, self.final_ops], feed_dict=td_map)
		if global_step % 100 == 0:
			self.writer.add_summary(summary, global_step)


	def save(self):
		print('Start Saving')
		for i in self.variable_parts2:
			i.saver.save(self.sess, './save/' + str(i.name), global_step=None, write_meta_graph=False, write_state=False)
		print('Saving Done.')



	# def inference(self, inputs):

	# 	vaerecon, z = self.sess.run([self.vae_loss.recon, self.z], feed_dict=self.place_holders.get_feed_dict_inference(inputs))

	# 	actions, values, states, neglogpacs = self.ppo.step(z, states)

	# 	return actions, values, states, neglogpacs, vaerecon

	# 	# log_probs = self.sess.run(self.log_probs, feed_dict=self.place_holders.get_feed_dict_inference(inputs))
	# 	# print(log_probs[0])
	# 	# def softmax(x):
	# 	# 	return np.exp(x) / np.sum(np.exp(x), axis=0)
		
	# 	# log_probs = softmax(log_probs[0])
	# 	# print(log_probs)

	# 	# action = np.random.choice(range(log_probs.shape[0]), p=log_probs.ravel())  # 根据概率来选 action
	# 	# return action
	# 	# z = self.sess.run(self.z, feed_dict=self.inputs.get_feed_dict_inference(inputs))

	# 	# self.mcts = MCTS(z, self.sess, self.policy_mcts, self.value_mcts, self.transition_mcts, self.z_mcts)

	# 	# return self.mcts.get_action()
