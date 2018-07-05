from persistence import Machine
import numpy as np
import yaml


from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl


BUFFER_LIMIT = 200
BATCH_SIZE = 8
KEEP_CNT = 10000

class Carla_Wrapper(object):

    def __init__(self):
        self.obs_buffer = []
        self.auxs_buffer = []
        self.control_buffer = []
        self.reward_buffer = []
        self.machine = Machine()
        self.global_step = 0
        self.update_cnt = 0

    def analyze_control(self, control):
        steer = control.steer
        throttle = control.throttle
        brake = control.brake
        hand_brake = control.hand_brake
        reverse = control.reverse
        print(steer, throttle, brake, hand_brake, reverse)

        b = 0
        if steer < -0.5:
            b += 2
        elif steer > 0.5:
            b += 1
        b *= 3

        if throttle > 0.5:
            b += 2
        if brake > 0.5:
            b += 1

        if reverse:
            b = 9
        
        if hand_brake:
            if steer < -0.5:
                b = 10
            elif steer > 0.5:
                b = 11
            else:
                b = 12
        
        return b


    def process_sensor_data(self, sensor_data):
        _main_image = sensor_data.get('CameraRGB', None)
        _mini_view_image1 = sensor_data.get('CameraDepth', None)
        _mini_view_image2 = sensor_data.get('CameraSemSeg', None)


        t1 = image_converter.to_rgb_array(_main_image)
        t2 = np.max(image_converter.depth_to_logarithmic_grayscale(_mini_view_image1), axis=2, keepdims=True)
        t3 = np.max(image_converter.to_rgb_array(_mini_view_image2), axis=2)
        return [t1, t2, t3]

    def update_all(self, measurements, sensor_data, control, reward):
        sensor_data = self.process_sensor_data(sensor_data)
        obs = [sensor_data[0]]
        auxs = [sensor_data[1], sensor_data[2],\
            measurements.player_measurements.forward_speed, \
            measurements.player_measurements.collision_vehicles + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_other,\
            measurements.player_measurements.intersection_otherlane + measurements.player_measurements.intersection_offroad]

        control = self.analyze_control(control)
        reward = [reward]
        
        assert len(self.obs_buffer) == len(self.auxs_buffer) and len(self.auxs_buffer) == len(self.control_buffer)\
             and len(self.control_buffer) == len(self.reward_buffer)

        tmp = 1
        for i in range(2, 500):
            tmp = tmp * 0.99
            if i < len(self.reward_buffer):
                self.reward_buffer[-i][0] += self.reward_buffer[-1][0] * tmp

        self.obs_buffer.append(obs)
        self.auxs_buffer.append(auxs)
        self.control_buffer.append(control)
        self.reward_buffer.append(reward)

        self.update_cnt += 1
        print(self.update_cnt)
        if self.update_cnt > BUFFER_LIMIT:
            print('Start Memory Replay')
            self.update_cnt = 0
            self.memory_training()
            print('Memory Replay Done')

        # self.red_buffer.append(red)
        # self.manual_buffer.append(manual)


    def memory_training(self):
        l = len(self.obs_buffer)
        fps = 10
        batch = []
        tmp_reward = np.array([i[0] for i in self.reward_buffer])
        tmp_reward = (tmp_reward - np.mean(tmp_reward)) / np.std(tmp_reward)

        for i in range(5, l - (fps // 2)):
            t = self.obs_buffer[i-16:i]
            if len(t) < 16:
                t = [self.obs_buffer[0]] * (16 - len(t)) + t

            t2 = self.obs_buffer[i - 16 + (fps // 2): i + (fps // 2)]
            if len(t2) < 16:
                t2 = [self.obs_buffer[0]] * (16 - len(t2)) + t2

            batch.append( [t, self.obs_buffer[i][0], self.auxs_buffer[i], self.control_buffer[i],\
                 tmp_reward[i], t2] )

        for i in range(0, len(batch), BATCH_SIZE):
            if i + BATCH_SIZE <= len(batch):
                self.machine.train(batch[i:i + BATCH_SIZE], self.global_step)
                self.global_step += 1

        if len(self.obs_buffer) > KEEP_CNT:
            self.obs_buffer = self.obs_buffer[1000:]
            self.auxs_buffer = self.auxs_buffer[1000:]
            self.control_buffer = self.control_buffer[1000:]
            self.reward_buffer = self.reward_buffer[1000:]


    def decode_control(self, cod):
        control = VehicleControl()

        control.steer = 0
        control.throttle = 0
        control.brake = 0
        control.hand_brake = False
        control.reverse = False


        if cod > 9:
            control.hand_brake = True
            if cod == 10:
                control.steer = -1
            elif cod == 10:
                control.steer = -1
            return control

        if cod == 9:
            control.reverse = True
            control.throttle = 1
            return control

        if cod % 3 == 1:
            control.brake = 1
        elif cod % 3 == 2:
            control.throttle = 1
        
        if cod // 3 == 1:
            control.steer = 1
        elif cod // 3 == 2:
            control.steer = -1
        
        return control

    def get_control(self, sensor_data):
        sensor_data = self.process_sensor_data(sensor_data)
        obs = [sensor_data[0]]

        t = self.obs_buffer[-15:]
        if len(t) == 0:
            t = [obs] * 15
        elif len(t) < 15:
            t = [self.obs_buffer[0]] * (15 - len(t)) + t

        control = self.machine.inference([[t + [obs]] for i in range(BATCH_SIZE)])

        control = self.decode_control(control)

        return control
