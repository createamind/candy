from persistence import Machine

import yaml

BUFFER_LIMIT = 350
BATCH_SIZE = 32
KEEP_CNT = 10000

class Carla_Wrapper(object):

    def __init__(self, machine):
        self.obs_buffer = []
        self.auxs_buffer = []
        self.control_buffer = []
        self.reward_buffer = []
        self.machine = machine
        self.global_step = 0
        self.update_cnt = 0

    def analyze_control(self.control):
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


    def update_all(self, measurements, sensor_data, control, reward):
        obs = [sensor_data.get('CameraRGB', None)]
        auxs = [sensor_data.get('CameraDepth', None), sensor_data.get('CameraSemSeg', None),\
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
            if i < len(self.reward_buffer)
                self.reward_buffer[-i][0] += self.reward_buffer[-1][0] * tmp

        self.obs_buffer.append(obs)
        self.auxs_buffer.append(auxs)
        self.control_buffer.append(control)
        self.reward_buffer.append(reward)

        self.update_cnt += 1
        if self.update_cnt > BUFFER_LIMIT:
            print('Start Memory Replay')
            self.update_cnt = 0
            self.memory_training()

        # self.red_buffer.append(red)
        # self.manual_buffer.append(manual)


    def memory_training(self):
        l = len(self.obs_buffer)
        fps = 10
        batch = []
        tmp_reward = np.array([i[0] for i in self.reward_buffer])
        tmp_reward = (tmp_reward - np.mean(tmp_reward)) / np.std(tmp_reward)

        for i in range(l + 5, l - (fps // 2)):
            batch.append( [self.obs_buffer[i-16:i], self.obs_buffer[i], self.auxs_buffer[i], self.control_buffer[i],\
                 tmp_reward[i], self.obs_buffer[i - 16 + (fps // 2): i + (fps // 2)]] )

        for i in range(0, len(batch), BATCH_SIZE):
            if i + BATCH_SIZE <= len(batch):
                self.machine.train(batch[i:i + BATCH_SIZE], self.global_step)
                self.global_step += 1

        if len(self.obs_buffer) > KEEP_CNT:
            self.obs_buffer = self.obs_buffer[1000:]
            self.auxs_buffer = self.auxs_buffer[1000:]
            self.control_buffer = self.control_buffer[1000:]
            self.reward_buffer = self.reward_buffer[1000:]

    def get_control(self, sensor_data):
        obs = [sensor_data.get('CameraRGB', None), sensor_data.get('Lidar32', None)]
        self.machine.inference(self.obs_buffer[-15:] + [obs])
