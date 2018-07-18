import tensorflow as tf
import os

class MLP:
	def __init__(self, args, name, x, s=10, hidden=100):
		self.args = args
		self.name = name
		self.x = x
		self.output_size = s
		self.hidden_size = hidden

	def inference(self):
		with tf.variable_scope(self.name) as _:
			logits = tf.layers.dropout(inputs=self.x, rate=0.2)
			logits = tf.layers.dense(inputs=logits, units=self.hidden_size, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
			
			logits = tf.nn.leaky_relu(logits) # Relu activation
			logits = tf.clip_by_value(logits, -5, 5)

			logits = tf.layers.dropout(inputs=logits, rate=0.2)
			self.outputs = tf.layers.dense(inputs=logits, units=self.output_size, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.args[self.name]['weight_decay']))
		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		return self.outputs


	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args[self.name]['learning_rate'])
		gvs = self.opt.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		# gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
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
	