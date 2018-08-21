# -*- coding: utf-8 -*-

from modules.c3d import C3D_Encoder

from modules.policy_gradient import PG, PGLoss

from modules.deconv import ImageDecoder

from modules.losses import MSELoss, CrossEntropyLoss

from modules.networks import MLP

from modules.vae import VAE, VAELoss

from modules.ppo import PPO, LstmPolicy

from modules.ppo2 import PPO2, Worker

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

		# z = tf.concat([z, self.speed], 1)
		# test_z = tf.concat([test_z, self.test_speed], 1)

		z = tf.clip_by_value(z, -5, 5)
		test_z = tf.clip_by_value(test_z, -5, 5)
		self.test_z = test_z
		# z = tf.Print(z, [z[0]], summarize=15)
		# test_z = tf.Print(test_z, [test_z[0]], summarize=20)

		self.ppo = PPO(args, 'ppo', z=z, test_z=test_z, ent_coef=0.00000001, vf_coef=1, max_grad_norm=0.5)

		self.ppo2 = PPO2(restore_weight=False)


		self.test_vae_loss.inference()
		# z = self.c3d_encoder.inference()

		self.variable_parts = [self.vae, self.ppo, self.test_vae]
		self.variable_parts2 = [self.vae, self.ppo]


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
		



	def step(self, obs, state):
		# mask = np.zeros(1)
		td_map = {self.ppo.act_model.S:state}
		td_map[self.test_raw_image] = np.array([obs[0]])# frame输入
		td_map[self.test_speed] = np.array([[obs[1]]])# speed

		return self.sess.run([self.ppo.act_model.a0], td_map)
		#return self.sess.run([self.ppo.act_model.a0, self.ppo.act_model.v0, self.ppo.act_model.snew, self.ppo.act_model.neglogp0, self.test_vae_loss.recon], td_map)

	def z_a_ppo2(self, obs, state):
		td_map = {}
		td_map[self.test_raw_image] = np.array([obs[0]])# frame输入
		z0 = self.sess.run(self.test_z,td_map)
		z = np.concatenate([z0,[[obs[1]]]],1)[0]
		return z, self.ppo2.choose_action(z)


	def update(self, training_data):
		self.ppo2.update(training_data)


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
		self.writer.add_summary(summary, global_step)


	def save(self):
		print('Start Saving')
		for i in self.variable_parts2:
			i.saver.save(self.sess, './save/' + str(i.name), global_step=None, write_meta_graph=False, write_state=False)
		print('Saving Done.')


