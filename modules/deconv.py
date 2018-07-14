import os
import random
import tensorflow as tf
import time
import numpy as np
import cv2


class DeconvNetwork:
	def __init__(self, args, name, depths=[1024, 512, 256, 128], s_size=7, last=3):
		self.args = args
		self.name = name
		self.depths = depths + [last]
		self.s_size = s_size

	def inference(self, inputs, name):

		inputs = tf.convert_to_tensor(inputs)
		with tf.variable_scope(name):
			# reshape from inputs
			with tf.variable_scope('reshape'):
				outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
				outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
				outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
				outputs = tf.clip_by_value(outputs, -5, 5)
			# deconvolution (transpose of convolution) x 4
			with tf.variable_scope('deconv1'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
				outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
				outputs = tf.clip_by_value(outputs, -5, 5)
			with tf.variable_scope('deconv2'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
				outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
				outputs = tf.clip_by_value(outputs, -5, 5)

			with tf.variable_scope('deconv3'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
				outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
				outputs = tf.clip_by_value(outputs, -5, 5)

			with tf.variable_scope('deconv4'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
			# output images
			with tf.variable_scope('tanh'):
				outputs = tf.tanh(outputs, name='outputs')
		return outputs

class ImageDecoder:
	def __init__(self, args, name, x, last=3):
		self.args = args
		self.name = name
		self.last = last
		self.x = x

	def inference(self):
		deconv = DeconvNetwork(self.args, self.name, last=self.last)
		self.output = deconv.inference(self.x, self.name)

		if self.name == 'raw_image':
			timage = tf.cast((self.output + 1) * 127, tf.uint8)
			tf.summary.image("raw_image", timage)

		# elif self.name == 'seg':
		# 	timage = tf.cast(tf.expand_dims(self.output, -1) * 20, tf.uint8)
		# 	tf.summary.image("seg", timage)

		elif self.name == 'depth':
			timage = tf.cast((self.output + 1) * 127, tf.uint8)
			tf.summary.image("depth", timage)

		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		return self.output


	# def optimize(self, loss):
	# 	self.opt = tf.train.AdamOptimizer(learning_rate=self.args[self.name]['learning_rate'])
	# 	opt_op = self.opt.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
	# 	return opt_op

	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args[self.name]['learning_rate'])
		gvs = self.opt.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		# print(self.name)
		# print(gvs)
		capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs if not grad is None]
		opt_op = self.opt.apply_gradients(capped_gvs)
		return opt_op

	def variable_restore(self, sess):

		model_filename = os.path.join("save", self.name)

		if os.path.isfile(model_filename + '.meta'):
			self.saver = tf.train.import_meta_graph(model_filename + '.meta')
			self.saver.restore(sess, model_filename)
			return

		if os.path.isfile(model_filename):
			self.saver.restore(sess, model_filename)
			return
		