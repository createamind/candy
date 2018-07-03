from persistence import Machine

BUFFER_LIMIT = 16

class Carla_Wrapper(object):

    def __init__(self, machine):
        self.obs_buffer = []
        self.auxs_buffer = []
        self.control_buffer = []
        self.reward_buffer = []
        self.red_buffer = []
        self.manual_buffer = []
        self.machine = machine

    def update_all(self, measurements, sensor_data, control, reward, red, manual):
        obs = [sensor_data.get('CameraRGB', None), sensor_data.get('Lidar32', None)]
        auxs = [sensor_data.get('CameraDepth', None), sensor_data.get('CameraSemSeg', None), measurements.player_measurements.forward_speed, \
            measurements.player_measurements.collision_vehicles, measurements.player_measurements.collision_pedestrians, measurements.player_measurements.collision_other,\
            measurements.player_measurements.intersection_otherlane, measurements.player_measurements.intersection_offroad]

        control = [control.steer, control.throttle, control.brake, control.hand_brake, control.reverse]
        reward = []
        assert len(self.obs_buffer) == len(self.auxs_buffer) and len(self.auxs_buffer) == len(self.control_buffer) and len(self.control_buffer) == len(self.reward_buffer) and len(self.reward_buffer) == len(self.red_buffer)
        self.obs_buffer.append(obs)
        self.auxs_buffer.append(auxs)
        self.control_buffer.append(control)
        self.reward_buffer.append(reward)
        self.red_buffer.append(red)
        self.manual_buffer.append(manual)


    def memory_training(self):
        self.machine.train([self.obs_buffer, self.auxs_buffer, self.control_buffer, self.reward_buffer, self.red_buffer, self.manual_buffer])
        self.obs_buffer = []
        self.auxs_buffer = []
        self.control_buffer = []
        self.reward_buffer = []
        self.red_buffer = []
        self.manual_buffer = []

    def get_control(self, sensor_data):
        obs = [sensor_data.get('CameraRGB', None), sensor_data.get('Lidar32', None)]
        self.machine.inference(obs, self.obs_buffer)
