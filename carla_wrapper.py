from persistence import Machine
import numpy as np
import yaml


from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from tqdm import tqdm
import msgpack
import msgpack_numpy as m
m.patch()

import os


BUFFER_LIMIT = 258
BATCH_SIZE = 128
KEEP_CNT = 258
TRAIN_EPOCH = 100

class Carla_Wrapper(object):

    def __init__(self):
        self.obs_buffer = []
        self.auxs_buffer = []
        self.control_buffer = []
        self.reward_buffer = []
        self.machine = Machine()
        self.global_step = 0
        self.update_cnt = 0
        self.pretrain()

    def analyze_control(self, control):
        steer = control.steer
        throttle = control.throttle
        brake = control.brake
        hand_brake = control.hand_brake
        reverse = control.reverse

        b = 0
        if steer < -0.2:
            b += 2
        elif steer > 0.2:
            b += 1
        b *= 3

        if throttle > 0.4:
            b += 2
        if brake > 0.4:
            b += 1

        if reverse:
            b = 9
        
        if hand_brake:
            if steer < -0.4:
                b = 10
            elif steer > 0.4:
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

    def update_all(self, measurements, sensor_data, control, reward, saveornot, collision):
        sensor_data = self.process_sensor_data(sensor_data)
        obs = [sensor_data[0]]
        auxs = [sensor_data[1], sensor_data[2],\
            abs(measurements.player_measurements.forward_speed) * 3.6 / 100, \
            collision / 20000,\
            measurements.player_measurements.intersection_offroad]

        control = self.analyze_control(control)
        reward = [reward]
        
        assert len(self.obs_buffer) == len(self.auxs_buffer) and len(self.auxs_buffer) == len(self.control_buffer)\
             and len(self.control_buffer) == len(self.reward_buffer)

        self.obs_buffer.append(obs)
        self.auxs_buffer.append(auxs)
        self.control_buffer.append(control)
        self.reward_buffer.append(reward)

        self.update_cnt += 1
        if self.update_cnt >= BUFFER_LIMIT:

            # Update BUFFER_LIMIT个
            last_one = self.reward_buffer[-1][0]
            l = len(self.reward_buffer)

            # if os.path.exists('obs/data'):
            #     with open('obs/data', 'rb') as fp:
            #         obs, auxs, control, reward = msgpack.load(fp, encoding='utf-8', raw=False)
            # else:
            #     obs, auxs, control, reward = [], [], [], []

            # with open('obs/data', 'wb') as fp:
            #     msgpack.dump([obs + self.obs_buffer[l - BUFFER_LIMIT:l], auxs + self.auxs_buffer[l - BUFFER_LIMIT:l], control + self.control_buffer[l - BUFFER_LIMIT:l], reward + self.reward_buffer[l - BUFFER_LIMIT:l]], fp, use_bin_type=True)
                

            for j in range(l - BUFFER_LIMIT, l):
                tmp = 1
                for i in range(1, 100):
                    tmp = tmp * 0.97
                    if i + j < len(self.reward_buffer):
                        self.reward_buffer[j][0] += self.reward_buffer[i + j][0] * tmp
                    else:
                        self.reward_buffer[j][0] += last_one * tmp

            print('Start Memory Replay')
            self.update_cnt = 0
            self.memory_training(saveornot)
            print('Memory Replay Done')
            return False

        return saveornot
        # self.red_buffer.append(red)
        # self.manual_buffer.append(manual)

    def pretrain(self):
        if os.path.exists('obs/data'):
            print('Start Pretraining!!')
            with open('obs/data', 'rb') as fp:
                self.obs_buffer, self.auxs_buffer, self.control_buffer, self.reward_buffer = msgpack.load(fp, encoding='utf-8', raw=False)
            print('Pretraining length = ', len(self.obs_buffer))
            self.memory_training(False, pretrain=True)

        
    def memory_training(self, saveornot, pretrain=False):
        l = len(self.obs_buffer)
        fps = 10
        batch = []
        tmp_reward = np.array([i[0] for i in self.reward_buffer])
        tmp_reward = (tmp_reward - np.mean(tmp_reward)) / np.std(tmp_reward)
        # for i in range(5, l - (fps // 2)):
        for i in range(l):
            # t = self.obs_buffer[i-16:i]
            # if len(t) < 16:
            #     t = [self.obs_buffer[0]] * (16 - len(t)) + t

            # t2 = self.obs_buffer[i - 16 + (fps // 2): i + (fps // 2)]
            # if len(t2) < 16:
            #     t2 = [self.obs_buffer[0]] * (16 - len(t2)) + t2

            batch.append( [None, self.obs_buffer[i][0], self.auxs_buffer[i], self.control_buffer[i],\
                 tmp_reward[i], None] )

        print("Memory Extraction Done.")

        for _ in tqdm(range(TRAIN_EPOCH)):
            for i in range(0, len(batch), BATCH_SIZE):
                if i + BATCH_SIZE <= len(batch):
                    self.machine.train(batch[i:i + BATCH_SIZE], self.global_step)
                    self.global_step += 1

                

        self.machine.save()

        if pretrain:            
            self.obs_buffer = []
            self.auxs_buffer = []
            self.control_buffer = []
            self.reward_buffer = []
            
        if len(self.obs_buffer) > KEEP_CNT + BUFFER_LIMIT:
        
            self.obs_buffer = self.obs_buffer[KEEP_CNT:]
            self.auxs_buffer = self.auxs_buffer[KEEP_CNT:]
            self.control_buffer = self.control_buffer[KEEP_CNT:]
            self.reward_buffer = self.reward_buffer[KEEP_CNT:]


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

        # t = self.obs_buffer[-15:]
        # if len(t) == 0:
        #     t = [obs] * 15
        # elif len(t) < 15:
        #     t = [self.obs_buffer[0]] * (15 - len(t)) + t

        control = self.machine.inference([ [None, obs[0]] for _ in range(BATCH_SIZE)])
        # control = self.machine.inference([ [t + [obs], obs[0]] for _ in range(BATCH_SIZE)])
        control = self.decode_control(control)
        return control
