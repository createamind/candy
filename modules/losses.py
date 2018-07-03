import tensorflow as tf

class MSELoss:

    def __init__(self, args, name, predict, label):
		self.args = args
		self.name = name

        self.predict = predict
        self.label = label

	def inference(self):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(self.label, self.predict))
        tf.summary.scalar(self.name, loss)

        return loss
    