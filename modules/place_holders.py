import tensorflow as tf

class Place_Holders(object):
    def __init__(self, args):
        self.args = args

        self.image_sequence = tf.placeholder(tf.float32, shape=(args.batch_size, 16, 112, 112, 3))

        self.raw_image = tf.placeholder(tf.float32, shape=(args.batch_size, 112, 112, 3))

        self.depth_image = tf.placeholder(tf.float32, shape=(args.batch_size, 112, 112, 1))

        self.seg_image = tf.placeholder(tf.int32, shape=(args.batch_size, 112, 112, 1))

        self.speed = tf.placeholder(tf.float32, shape=(args.batch_size, 1))

        self.collision = tf.placeholder(tf.float32, shape=(args.batch_size, 1))

        self.intersection = tf.placeholder(tf.float32, shape=(args.batch_size, 1))

        self.control = tf.placeholder(tf.float32, shape=(args.batch_size, 5))

        self.reward = tf.placeholder(tf.float32, shape=(args.batch_size, 4))

        self.next_image = tf.placeholder(tf.float32, shape=(args.batch_size, 112, 112, 3))

    def inference(self):

        return [self.image_sequence, self.raw_image, self.depth_image, self.seg_image, self.speed, self.collision, self.intersection, self.control, self.reward, self.next_image]



## [self.obs_buffer, self.auxs_buffer, self.control_buffer, self.reward_buffer, self.red_buffer, self.manual_buffer]
    def get_feed_dict_train(self, inputs):
        return {
            self.image_sequence: self._get_image_sequence(inputs[0])
            self.raw_image: None
            }

    def _get_image_sequence(self, obs):
        to_return = []
        blank_picture = None
        for i in obs:
            found_sequence = []
            for j in range(i - 15, i + 1):
                if j < 0:
                    found_sequence.append(blank_picture)
                else:
                    found_sequence.append(obs[j])
            to_return.append(found_sequence)

        return to_return



    def get_feed_dict_inference(self, inputs):
        return {
            self.image_sequence: self._get_image_sequence(inputs)
            }

