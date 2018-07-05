import os
import random
import tensorflow as tf
import time
import numpy as np
import cv2


class DeconvNetwork:
	def __init__(self, depths=[1024, 512, 256, 128], s_size=7, last=3):
		self.depths = depths + [last]
		self.s_size = s_size

	def inference(self, inputs, name):

		inputs = tf.convert_to_tensor(inputs)
		with tf.variable_scope(name):
			# reshape from inputs
			with tf.variable_scope('reshape'):
				outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
				outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
				outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
			# deconvolution (transpose of convolution) x 4
			with tf.variable_scope('deconv1'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
				outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
			with tf.variable_scope('deconv2'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
				outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
			with tf.variable_scope('deconv3'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
				outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
			with tf.variable_scope('deconv4'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
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
		deconv = DeconvNetwork(last=self.last)
		self.output = deconv.inference(self.x, self.name)
		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		return self.output


	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args[self.name]['learning_rate'])
		opt_op = self.opt.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		return opt_op


	def variable_restore(self, sess):

		model_filename = os.path.join("save", self.name)
		if os.path.isfile(model_filename):
			self.saver.restore(sess, model_filename)
		