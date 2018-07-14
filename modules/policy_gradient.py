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
		capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
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

	

class PGLoss:
	def __init__(self, args, name, action, reward, log_probs):
		self.args = args
		self.name = name
		self.action = action
		self.reward = reward
		self.log_probs = log_probs

		
	def inference(self):
		neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.log_probs, labels=self.action)
		# self.log_probs = tf.Print(self.log_probs, [self.log_probs])
		loss = tf.reduce_mean(neg_log_prob * self.reward)
		
		probs = tf.nn.softmax(self.log_probs)

		entropy = tf.reduce_mean(tf.reduce_mean(- probs * tf.log(tf.clip_by_value(probs,1e-10,1.0)), 1), 0)
		

		loss = tf.Print(loss, [loss])
		entropy = tf.Print(entropy, [entropy])

		totloss = loss - 10 * entropy

		tf.summary.scalar(self.name, totloss)
		return totloss

