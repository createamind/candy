from scipy.misc import imsave

import sys
import tensorflow as tf
import numpy as np
import math
import os

class VAE():
	def __init__(self, args, name, x, reuse=False):
		self.args = args
		self.name = name
		self.x = x
		self.reuse = reuse

	def encoder(self, x):
		"""Define q(z|x) network"""

		with tf.variable_scope(self.name, reuse=self.reuse) as _:
			with tf.variable_scope('encoder', reuse=self.reuse) as _2:

				x = tf.nn.relu(tf.layers.conv2d(x, 16, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d(x, 256, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d(x, 512, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				
				x = tf.reshape(x, [-1, 51200])
				z = tf.layers.dense(x, 30, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
				
				mean, logsigma = tf.split(z, 2, 1)

		return mean, logsigma


	def decoder(self, mean, logsigma):
		sigma = tf.exp(logsigma)
		eps = tf.random_normal(tf.shape(sigma))
		x = sigma * eps + mean

		with tf.variable_scope(self.name, reuse=self.reuse) as _:
			with tf.variable_scope('decoder', reuse=self.reuse) as _2:

				x = tf.layers.dense(x, 51200, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
				x = tf.reshape(x, [-1, 10, 10, 512])

				x = tf.nn.relu(tf.layers.conv2d_transpose(x, 256, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d_transpose(x, 128, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d_transpose(x, 64, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d_transpose(x, 16, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.tanh(tf.layers.conv2d_transpose(x, 8, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
		
		return x

	def inference(self):

		timage = tf.cast((self.x + 1) * 127, tf.uint8)
		tf.summary.image("raw_image_real", timage[:,:,:,:3])
		# tf.summary.image("raw_image_real", timage[:,:,:,4:])

		mean, logsigma = self.encoder(self.x)
		recon_x = self.decoder(mean, logsigma)

		timage = tf.cast((recon_x + 1) * 127, tf.uint8)
		tf.summary.image("raw_image_recon", timage[:,:,:,:3])
		# tf.summary.image("raw_image_recon", timage[:,:,:,4:])

		
		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

		return recon_x, mean, logsigma

	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args[self.name]['learning_rate'])
		gvs = self.opt.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		# gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs if not grad is None]
		opt_op = self.opt.apply_gradients(gvs)
		return opt_op


	def variable_restore(self, sess):

		model_filename = os.path.join("save", self.name)

		# if os.path.isfile(model_filename + '.meta'):
		# 	self.saver = tf.train.import_meta_graph(model_filename + '.meta')
		# 	self.saver.restore(sess, model_filename)
		# 	return

		if os.path.isfile(model_filename + '.data-00000-of-00001'):
			self.saver.restore(sess, model_filename)
			return
	
class VAELoss():


	def __init__(self, args, name, recon_x, x, mu, logsigma):
		self.args = args
		self.name = name
		self.recon_x = recon_x
		self.x = x
		self.mu = mu
		self.logsigma = logsigma

	def inference(self):
		const = 1 / (self.args['batch_size'] * self.args['x_dim'] * self.args['y_dim'])
		
		self.recon = const * tf.reduce_sum(tf.squared_difference(self.x, self.recon_x))
		self.vae = const * -0.5 * tf.reduce_sum(1.0 + 2.0 * self.logsigma - tf.square(self.mu) - tf.exp(2 * self.logsigma))
		
		tf.summary.scalar(self.name + 'loss_vae', self.vae)
		tf.summary.scalar(self.name + 'loss_recon', self.recon)

		return tf.reduce_sum(self.recon + 100 * self.vae)




class VAEVisualize():
	def __init__(self, args, name, x):
		self.args = args
		self.name = name
		self.x = x

	def encoder(self, x):
		"""Define q(z|x) network"""

		with tf.variable_scope(self.name) as _:
			with tf.variable_scope('encoder') as _2:

				x = tf.nn.relu(tf.layers.conv2d(x, 16, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d(x, 256, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				
				x = tf.reshape(x, [-1, 51200])
				z = tf.layers.dense(x, 30, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
				
				mean, logsigma = tf.split(z, 2, 1)

		return mean, logsigma


	def decoder(self, x):

		with tf.variable_scope(self.name) as _:
			with tf.variable_scope('decoder') as _2:

				x = tf.layers.dense(x, 51200, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
				x = tf.reshape(x, [-1, 10, 10, 512])

				x = tf.nn.relu(tf.layers.conv2d_transpose(x, 256, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d_transpose(x, 128, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d_transpose(x, 64, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.relu(tf.layers.conv2d_transpose(x, 16, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))
				x = tf.nn.tanh(tf.layers.conv2d_transpose(x, 3, [4, 4], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay'])))

		return x


	def inference(self):

		recon_x = self.decoder(self.x)

		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

		return recon_x

	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args[self.name]['learning_rate'])
		gvs = self.opt.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		# gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs if not grad is None]
		opt_op = self.opt.apply_gradients(gvs)
		return opt_op


	def variable_restore(self, sess):

		model_filename = os.path.join("save", self.name)

		# if os.path.isfile(model_filename + '.meta'):
		# 	self.saver = tf.train.import_meta_graph(model_filename + '.meta')
		# 	self.saver.restore(sess, model_filename)
		# 	return

		if os.path.isfile(model_filename + '.data-00000-of-00001'):
			self.saver.restore(sess, model_filename)
			return