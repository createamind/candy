from modules.c3d import C3D_Encoder

from modules.policy_gradient import PG, PGLoss
from modules.place_holders import PlaceHolders

from modules.deconv import ImageDecoder

from modules.losses import MSELoss, CrossEntropyLoss

from modules.networks import MLP

from modules.vae import VAE, VAELoss, VAEVisualize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
		self.args['crop_size'] = 320
		self.args['num_frames_per_clip'] = 16
		
		#Building Graph
		# self.place_holders = PlaceHolders(args)

		# inputs = self.place_holders.inference()
		#[self.image_sequence, self.raw_image, self.depth_image, self.seg_image, self.speed, self.collision, self.intersection, self.control, self.reward, self.transition]

		# self.c3d_encoder = C3D_Encoder(args,'c3d_encoder', inputs[0])
		# self.c3d_future = C3D_Encoder(args,'c3d_encoder', inputs[9], reuse=True)

		self.int_z = tf.placeholder(tf.float32, shape=(20, 15))
		self.vae_vi = VAEVisualize(args, 'vae', self.int_z)
		self.images = self.vae_vi.inference()

		self.images = tf.cast((self.images + 1) * 127, tf.uint8)
		# timage = tf.cast((self.images + 1) * 127, tf.uint8)
		# tf.summary.image("visualization", timage)
		
		# self.vae = VAE(args, 'vae', inputs[1])
		# self.future_vae = VAE(args, self.c3d_future.inference())

		# recon_x, z, logsigma = self.vae.inference()
		# z = tf.Print(z, [z])
		# self.z = z
		# self.vae_loss = VAELoss(args, 'vae', recon_x, inputs[1], z, logsigma)


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
		self.variable_parts = [self.vae_vi]
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
		# self.loss_parts = self.vae_loss.inference()
		# self.loss_parts = self.raw_decoder_loss.inference()
				
		# weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses'))
		# tf.summary.scalar('weight_decay_loss', weight_decay_loss)
		# total_loss = self.loss_parts
		# tf.summary.scalar('total_loss', tf.reduce_mean(total_loss))

		# self.final_ops = []
		# for part in self.variable_parts:
		# 	self.final_ops.append(part.optimize(total_loss))
		# self.final_ops = tf.group(self.final_ops)

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
		


	def train(self, inputs, global_step):
		summary, _ = self.sess.run([self.merged, self.final_ops], feed_dict=self.place_holders.get_feed_dict_train(inputs))
		self.writer.add_summary(summary, global_step)


	def save(self):
		print('Start Saving')
		for i in self.variable_parts:
			i.saver.save(self.sess, './save/' + str(i.name), global_step=None)
		print('Saving Done.')



	def inference(self, inputs):
		log_probs = self.sess.run(self.log_probs, feed_dict=self.place_holders.get_feed_dict_inference(inputs))
		print(log_probs[0])
		def softmax(x):
			return np.exp(x) / np.sum(np.exp(x), axis=0)
		
		log_probs = softmax(log_probs[0])
		print(log_probs)

		action = np.random.choice(range(log_probs.shape[0]), p=log_probs.ravel())  # 根据概率来选 action
		return action
		# z = self.sess.run(self.z, feed_dict=self.inputs.get_feed_dict_inference(inputs))

		# self.mcts = MCTS(z, self.sess, self.policy_mcts, self.value_mcts, self.transition_mcts, self.z_mcts)

		# return self.mcts.get_action()


def main():
	# z = [-0.977040291, 1.63333166, 0.507133842, -0.733503461, 1.18871117, -0.720566, -1.49697387, -0.501638234, -0.696030438, -0.371694297, -0.600599408, 0.0596709624, -1.01534212, 0.583785832, -0.269926041, 0.337309062, 0.307009071, 0.554181635, 0.550192237, -0.953083575, -0.0576407611, 0.00375251658, 0.0730978549, 1.19308269, 0.424325794, -0.495179415, 0.537171483, -0.457180202, -1.11410332, 1.52096176, -0.695364416, 1.62396383, 0.206000328, -0.150249943, 0.665629387, 0.311423302, 0.374389052, -0.86861521, -1.13608301, 0.027664993, 0.415747732, -1.17723179, -1.24014664, -0.216029346, -0.258248836, 1.26021576, 0.400671601, -1.66847956, -0.658004701, -1.07025611]
	# z = [-0.480454236, 0.953269243, 0.330706298, 0.0310323909, 0.633806229, -0.0643456876, -0.174652278, 1.12146568, 0.831449032, 0.451472759, -1.99872589, 1.26931345, -1.11463642, 0.335262537, 0.0950239375, 0.61537385, -0.187405884, -0.127848133, -1.34532118, -0.0804636106, 0.199894339, -0.938853264, 0.329392225, -0.0442294776, 0.695968747, 0.519495487, -0.234864563, 0.267519474, 0.694029, -0.726179063, -0.369746059, -0.102801062, -0.569598615, 0.99882412, 0.563601494, 0.125620246, 0.0969938338, 1.8117075, 0.299527168, -0.747712731, 1.35097492, -0.026486624, -0.481032729, -0.355826408, -0.0545232445, 0.771909714, 0.363240123, 0.859262228, 0.684637904, -0.192559198]
	# z = [-0.0235790908, 0.0780012831, 1.31324303, -0.588644326, 0.515748]
	z = [-0.52, -2.31, 1.75, -1.28, -1.04, 0.09, -0.26, -3.11, 1.49, -0.14, -2.13, 2.27, -0.52, 0.40, 0.16]
	z = np.array(z)
	machine = Machine()


	F50 = 15
	F20 = 20
	all_images = []
	for i in range(F50):
		feed = []
		tmpz = np.copy(z)
		for j in list(np.linspace(-1.5, 1.5, F20)):
			tmpz[i] = j
			feed.append(np.copy(tmpz))
		images = machine.sess.run(machine.images, feed_dict={machine.int_z: np.array(feed)})
		for j in range(F20):
			image = images[j]
			fig = plt.figure(1)
			ax = fig.add_subplot(F50, F20, i * F20 + j + 1)
			ax.get_yaxis().set_visible(False)
			ax.get_xaxis().set_visible(False)
			# print(np.array(np.transpose(image, (1,2,0)).shape))
			ax.imshow(image[:,:,:3])
	
	plt.subplots_adjust(wspace=0.0001, hspace=0.0001)
	plt.savefig('visualize.png', dpi=1000)
	# plt.show()

	

if __name__ == '__main__':
	main()