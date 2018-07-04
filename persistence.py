from modules.c3d import C3D_Encoder
from modules.place_holders import PlaceHolders

from modules.vae import VAE

from modules.mcts import MCTS

from modules.deconv import ImageDecoder

from modules.losses import MSELoss

from modules.networks import MLP, TransitionNetwork, PolicyNetwork, ValueNetwork

import tensorflow as tf

import yaml


class Machine(object):
    def __init__(self):

        self.args = self.get_args()
        self.args.crop_size = 112
		self.args.num_frames_per_clip = 16
		self.batch_size = 10
        
        #Building Graph
        self.place_holders = PlaceHolders(args)

        inputs = self.place_holders.inference()
        #[self.image_sequence, self.raw_image, self.depth_image, self.seg_image, self.speed, self.collision, self.intersection, self.control, self.reward, self.transition]

        self.c3d_encoder = C3D_Encoder(args,'c3d_encoder', inputs[0])
        self.c3d_future = C3D_Encoder(args,'c3d_encoder', inputs[9])

        # self.vae = VAE(args, self.c3d_encoder.inference())
        # self.future_vae = VAE(args, self.c3d_future.inference())


        z = self.c3d_encoder.inference()
        self.z = z

        self.raw_decoder = ImageDecoder(args, 'raw_image', z)
        self.raw_decoder_loss = MSELoss(args, 'raw_image', self.raw_decoder.inference(), inputs[1])

        self.seg_decoder = ImageDecoder(args, 'seg', z)
        self.seg_decoder_loss = MSELoss(args, 'seg', self.seg_decoder.inference(), inputs[3])
        
        self.depth_decoder = ImageDecoder(args, 'depth', z)
        self.depth_decoder_loss = MSELoss(args, 'depth', self.depth_decoder.inference(), inputs[2])

        self.speed_prediction = MLP(args, 'speed', z)
        self.speed_loss = MSELoss(args, self.speed_prediction.inference(), inputs[4])        

        self.collision_prediction = MLP(args, 'collision', z)
        self.collision_loss = MSELoss(args, self.collision_prediction.inference(), inputs[5])

        self.intersection_prediction = MLP(args, 'intersection', z)
        self.intersection_loss = MSELoss(args, self.intersection_prediction.inference(), inputs[6])


        self.policy = MLP(args, 'policy', z, 5, 100)
        self.value = MLP(args, 'value', z, 4, 100)
        self.transition = MLP(args, 'transition', z, 100, 100)

        self.imitation_loss = MESLoss(args, self.policy.inference(), inputs[7])
        self.reward_loss = MESLoss(args, self.value.inference(), inputs[8])
        self.transition_loss = MESLoss(args, self.transition.inference(), self.future_vae.inference())


        # MCTS
        self.z_mcts = tf.placeholder(tf.float32, shape=(1, 100))
        self.policy_mcts = MLP(args, 'policy', self.z_mcts, 5, 100).inference()
        self.value_mcts = MLP(args, 'value', self.z_mcts, 4, 100).inference()
        self.transition_mcts = MLP(args, 'transition', self.z_mcts, 100, 100).inference()

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

        self.variable_parts = [self.c3d_encoder, self.raw_decoder, self.seg_decoder, self.policy]
        self.loss_parts = self.collision_loss.inference() + self.intersection_loss.inference() + self.speed_loss.inference() + \
                self.raw_decoder_loss.inference() + self.seg_decoder_loss.inference() + self.imitation_loss.inference() + self.reward_loss.inference() + self.transition_loss.inference()
        
        
        weight_decay_loss = tf.get_collection('weightdecay_losses')
        vae_loss = tf.get_collection('vae_loss')
        total_loss = tf.add_n(self.loss_parts) + weight_decay_loss + vae_loss
        tf.summary.scalar('total_loss', tf.reduce_mean(total_loss))

        self.final_ops = []
        for part in self.variable_parts:
            self.final_ops.append(part.optimize(total_loss))
        self.final_ops = tf.group(self.final_ops)

        config = tf.ConfigProto(allow_soft_placement = True)
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())

        for part in self.variable_parts:
            part.variable_restore(self.sess, checkpoint=-1)

        # Create summary writter
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/candy', sess.graph)

        print('Model Started!')

    def get_args(self):
        with open("args.yaml", 'r') as f:
            try:
                t = yaml.load(f)
                print(t)
                return t
            except yaml.YAMLError as exc:
                print(exc)


    def train(self, inputs):
        for step in xrange(10):
            self.sess.run(self.final_ops, feed_dict=self.inputs.get_feed_dict_train(inputs))


    def inference(self, inputs):
        self.sess.run(self.z, feed_dict=self.inputs.get_feed_dict_inference(inputs))

        self.mcts = MCTS(self.sess, self.policy_mcts, self.value_mcts, self.transition_mcts)

        return self.mcts.get_action()
