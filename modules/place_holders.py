import tensorflow as tf
import numpy as np

class PlaceHolders(object):
    def __init__(self, args):
        self.args = args

        self.image_sequence = tf.placeholder(tf.float32, shape=(args['batch_size'], 16, 112, 112, 3))

        self.raw_image = tf.placeholder(tf.float32, shape=(args['batch_size'], 112, 112, 3))

        self.depth_image = tf.placeholder(tf.float32, shape=(args['batch_size'], 112, 112, 1))

        self.seg_image = tf.placeholder(tf.int32, shape=(args['batch_size'], 112, 112))

        self.speed = tf.placeholder(tf.float32, shape=(args['batch_size'], 1))

        self.collision = tf.placeholder(tf.float32, shape=(args['batch_size'], 1))

        self.intersection = tf.placeholder(tf.float32, shape=(args['batch_size'], 1))

        self.control = tf.placeholder(tf.int32, shape=(args['batch_size']))

        self.reward = tf.placeholder(tf.float32, shape=(args['batch_size']))

        self.next_sequence = tf.placeholder(tf.float32, shape=(args['batch_size'], 16, 112, 112, 3))

    def inference(self):

        return [self.image_sequence, self.raw_image, self.depth_image, self.seg_image,\
         self.speed, self.collision, self.intersection, self.control, self.reward, self.next_sequence]


        # obs = [sensor_data.get('CameraRGB', None)]
        # auxs = [sensor_data.get('CameraDepth', None), sensor_data.get('CameraSemSeg', None),\
        #     measurements.player_measurements.forward_speed, \
        #     measurements.player_measurements.collision_vehicles + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_other,\
        #     measurements.player_measurements.intersection_otherlane + measurements.player_measurements.intersection_offroad]

        # control = [control.steer, control.throttle, control.brake, control.hand_brake, control.reverse]
        # reward = [reward]
#   batch.append( [self.obs_buffer[i-16:i], self.obs_buffer[i], self.auxs_buffer[i], self.control_buffer[i],\
#   self.reward_buffer[i], self.obs_buffer[i - 16 + (fps // 2): i + (fps // 2)]] )
   
   
    def get_feed_dict_train(self, inputs):
        my_dict =  {
            self.image_sequence: self._process_image_sequence([v[0] for v in inputs]),
            self.raw_image: self._process_image([v[1] for v in inputs], 'raw'),
            self.depth_image: self._process_image([v[2][0] for v in inputs], 'depth'),
            self.seg_image: self._process_image([v[2][1] for v in inputs], 'seg'),

            self.speed: [[v[2][2]] for v in inputs],
            self.collision: [[v[2][3]] for v in inputs],
            self.intersection: [[v[2][4]] for v in inputs],

            self.control: [v[3] for v in inputs],
            self.reward: [v[4] for v in inputs],

            self.next_sequence: self._process_image_sequence([v[5] for v in inputs])
        }
        return my_dict


    def _process_image_sequence(self, image_sequence):
        t = self._process_image([v[0] for batch in image_sequence for v in batch], 'raw')
        # print(t[0])
        # print('-' * 40)
        # print(t[1])
        return np.reshape(t, [len(image_sequence), 16, 112, 112, 3])
        # May be not 16
        # [[[rgb],[rgb],[rgb],[rgb]], []]
        # to (args['batch_size'], 16, 112, 112, 3)

    def _process_image(self, image, typeofimage):

        if typeofimage == 'raw':
            image = np.array(image).astype(float) / 255
            return image
        else:
            return image
        return image

    def get_feed_dict_inference(self, inputs):
        return {
            self.image_sequence: self._process_image_sequence([v[0] for v in inputs]),
        }

