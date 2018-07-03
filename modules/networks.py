class MLP:
	def __init__(self, args, name, x, s=10, hidden=100):
		self.args = args
		self.name = name
        self.x = x
        self.output_size = s
        self.hidden_size = hidden

	def inference(self):
        with variable_scope(self.name) as var_scope:
            logits = tf.layers.dropout(inputs=logits, rate=0.5)
            logits = tf.layers.dense(inputs=self.x, units=self.hidden_size, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            logits = tf.layers.dropout(inputs=logits, rate=0.5)
            self.outputs = tf.layers.dense(inputs=self.x, units=self.output_size, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        return self.outputs


	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate[self.name])
		opt_op = self.opt.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		return opt_op


	def variable_restore(self, sess):

		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
		model_filename = os.path.join("save", self.name)
		if os.path.isfile(model_filename):
			self.saver.restore(sess, model_filename)
    