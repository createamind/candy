from persistence import Machine

BUFFER_LIMIT = 350
BATCH_SIZE = 32

class Carla_Wrapper(object):

    def __init__(self, machine):
        self.obs_buffer = []
        self.auxs_buffer = []
        self.control_buffer = []
        self.reward_buffer = []
        self.machine = machine

    def update_all(self, measurements, sensor_data, control, reward):
        obs = [sensor_data.get('CameraRGB', None), sensor_data.get('Lidar32', None)]
        auxs = [sensor_data.get('CameraDepth', None), sensor_data.get('CameraSemSeg', None),\
            measurements.player_measurements.forward_speed, measurements.player_measurements.collision_vehicles,\
            measurements.player_measurements.collision_pedestrians, measurements.player_measurements.collision_other,\
            measurements.player_measurements.intersection_otherlane, measurements.player_measurements.intersection_offroad]

        control = [control.steer, control.throttle, control.brake, control.hand_brake, control.reverse]

        assert len(self.obs_buffer) == len(self.auxs_buffer) and len(self.auxs_buffer) == len(self.control_buffer)\
             and len(self.control_buffer) == len(self.reward_buffer)

        self.obs_buffer.append(obs)
        self.auxs_buffer.append(auxs)
        self.control_buffer.append(control)
        self.reward_buffer.append(reward)


        if len(self.obs_buffer) > BUFFER_LIMIT:
            print('Start Memory Replay')
            self.memory_training()

        # self.red_buffer.append(red)
        # self.manual_buffer.append(manual)


    def memory_training(self):
        l = len(self.obs_buffer)
        fps = 10
        batch = []
        for i in range(l + 5, l - (fps // 2)):
            batch.append( [self.obs_buffer[i-16:i], self.obs_buffer[i], self.auxs_buffer[i], self.control_buffer[i],\
                 self.reward_buffer[i], self.obs_buffer[i - 16 + (fps // 2): i + (fps // 2)]] )

        for i in range(0, len(batch), BATCH_SIZE):
            if i + BATCH_SIZE <= len(batch):
                self.machine.train(batch[i:i + BATCH_SIZE])

        self.obs_buffer = []
        self.auxs_buffer = []
        self.control_buffer = []
        self.reward_buffer = []

    def get_control(self, sensor_data):
        obs = [sensor_data.get('CameraRGB', None), sensor_data.get('Lidar32', None)]
        self.machine.inference(self.obs_buffer[-15:] + [obs])
