import os
import tensorflow as tf

class PG:
	def __init__(self, args, name, x, output=10, hidden=100):
		self.args = args
		self.name = name
		self.x = x
		self.output_size = output
		self.hidden_size = hidden

	def inference(self):
		with tf.variable_scope(self.name) as var_scope:
			logits = tf.layers.dropout(inputs=self.x, rate=0.5)
			logits = tf.layers.dense(inputs=logits, units=self.hidden_size, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
			logits = tf.layers.dropout(inputs=logits, rate=0.5)
			self.outputs = tf.layers.dense(inputs=logits, units=self.output_size, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		return self.outputs


	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args[self.name]['learning_rate'])
		opt_op = self.opt.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		return opt_op


	def variable_restore(self, sess):

		model_filename = os.path.join("save", self.name)
		if os.path.isfile(model_filename):
			self.saver.restore(sess, model_filename)
	

class PGLoss:
	def __init__(self, args, name, action, reward, log_probs):
		self.args = args
		self.name = name
		self.action = action
		self.reward = reward
		self.log_probs = log_probs

		
	def inference(self):
		neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.log_probs, labels=self.action)
		loss = tf.reduce_mean(neg_log_prob * self.reward)
		
		tf.summary.scalar(self.name, loss)
		return loss

