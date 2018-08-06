#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    R            : restart level
    1-9            : set reward
    n            : disable imitation learning
    m            : enable imitation learning(default)
    t            : toggle display
    c            : toggle model control(Default: model control the car. You may change it into manual control)
    v            : reset now

STARTING in a moment...
"""

from __future__ import print_function, absolute_import, division

import argparse
import logging
import random
import time
import datetime
try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_t
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_1
    from pygame.locals import K_2
    from pygame.locals import K_3
    from pygame.locals import K_4
    from pygame.locals import K_5
    from pygame.locals import K_6
    from pygame.locals import K_7
    from pygame.locals import K_8
    from pygame.locals import K_9
    from pygame.locals import K_v
    from pygame.locals import K_c

    
    
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from carla_wrapper import Carla_Wrapper

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
MINI_WINDOW_WIDTH = 200
MINI_WINDOW_HEIGHT = 200
BUFFER_LIMIT = 100

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=25,
        NumberOfPedestrians=50,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(320, 320)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(320, 320)
    camera1.set_position(2.0, 0.0, 1.4)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
    camera2.set_image_size(320, 320)
    camera2.set_position(2.0, 0.0, 1.4)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)

    camera3 = sensor.Camera('CameraForHuman')
    camera3.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera3.set_position(2.0, 0.0, 1.4)
    camera3.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera3)

    if args.lidar:
        lidar = sensor.Lidar('Lidar32')
        lidar.set_position(0, 0, 2.5)
        lidar.set_rotation(0, 0, 0)
        lidar.set(
            Channels=32,
            Range=50,
            PointsPerSecond=100000,
            RotationFrequency=10,
            UpperFovLimit=10,
            LowerFovLimit=-30)
        settings.add_sensor(lidar)
    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args, wrapper):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = True
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        self.should_display = True
        random.seed(datetime.datetime.now())
        self.manual = True
        self.manual_control = (random.randint(1,1000) == 1)
        self.cnt = 0 #与buffer_limit进行比较，控制训练的切换
        self.history_collision = 0
        self.ucnt = 0
        self.prev_control = None
        self.history_steer = 0
        self.endnow = False#按下v会置为True，立刻进行ｔｒａｉｎｉｎｇ
    

        self.carla_wrapper = wrapper

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop() #整个模型与carla交互.
                self._on_render() #pygame展示
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        # self._carla_settings.set(SeedVehicles=3)
        # self._carla_settings.set(SeedPedestrians=3)
        self._carla_settings.set(WeatherId=0)

        # self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        # player_start = 59
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        self._main_image = sensor_data.get('CameraForHuman', None)
        self._mini_view_image1 = sensor_data.get('CameraDepth', None)
        self._mini_view_image2 = sensor_data.get('CameraSemSeg', None)
        self._lidar_measurement = sensor_data.get('Lidar32', None)
        # self._human_image = sensor_data.get('CameraForHuman', None)


        collision = self.get_collision(measurements) #得到瞬时的碰撞
        control, manual_reward = self._get_keyboard_control(pygame.key.get_pressed()) #得到键盘的输入，以及键盘输入的reward
        reward, _ = self.calculate_reward(measurements, manual_reward, collision) #计算reward，如果键盘没有reward


        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation, reward)
            else:
                self._print_player_measurements(measurements.player_measurements)

            # Plot position on the map as well.

            self._timer.lap()


        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents


        if control == "done":
            control = measurements.player_measurements.autopilot_control
            self.history_steer = control.steer
            self.client.send_control(control)
            return
        elif control is None:
            self._on_new_episode()
            return

        # if self.prev_control is None or self.ucnt == 5:

        if self._enable_autopilot: #系统自带的自动驾驶
            control = measurements.player_measurements.autopilot_control


        model_control = self.carla_wrapper.get_control([measurements, sensor_data, control, reward, self.history_steer, control, self.manual])


        # else:
        #     if random.randint(1,2) == 1:
        #     else:
        #         control = measurements.player_measurements.autopilot_control


        #     self.ucnt = 0
        #     self.prev_control = control
        # else:
        #     self.ucnt += 1
        #     control = self.prev_control

        # print(measurements.player_measurements.transform.rotation)
        # print(measurements.player_measurements.transform.location)
        # print(measurements.player_measurements.transform.orientation)
        print(control)
        print(model_control)

        #Speed Limit
        if measurements.player_measurements.forward_speed * 3.6 > 30:
            model_control.throttle = 0

        if self.manual_control:
            self.history_steer = control.steer
            self.client.send_control(control)
        else:
            self.history_steer = model_control.steer
            self.client.send_control(model_control)

        #控制什么时候进行ｔｒａｉｎｉｎｇ
        if self.endnow or (self.cnt > 5 and (self.cnt > BUFFER_LIMIT or collision > 0 or measurements.player_measurements.intersection_offroad > 0.5\
         or measurements.player_measurements.intersection_otherlane > 0.5)):
        # if self.endnow or (self.cnt > 10 and (self.cnt > BUFFER_LIMIT or collision > 0)):
            #总结这段时间的情况，调用training
            rewardlala = -1 if (collision > 0 or measurements.player_measurements.intersection_offroad > 0.5 or measurements.player_measurements.intersection_otherlane > 0.5) else None
            self.carla_wrapper.post_process([measurements, sensor_data, model_control, rewardlala, self.history_steer, control, self.manual], self.cnt)
            self.cnt = 0
            self.endnow = False
        else:
            #不应该进行trainging
            self.cnt += 1
            self.endnow = False
            #记录当前步的状况
            self.carla_wrapper.update([measurements, sensor_data, model_control, reward, self.history_steer, control, self.manual])


    def get_collision(self, measurements):
        new_collision = measurements.player_measurements.collision_vehicles + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_other
        ans = new_collision - self.history_collision #得到瞬时值 累加值的差
        self.history_collision = new_collision
        return ans

    
    def calculate_reward(self, measurements, reward, collision):
        
        speed = measurements.player_measurements.forward_speed * 3.6 if measurements.player_measurements.forward_speed > 0 else 0
        
        # collision = measurements.player_measurements.collision_vehicles + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_other

        # intersection = measurements.player_measurements.intersection_otherlane + measurements.player_measurements.intersection_offroad
        
        intersection = measurements.player_measurements.intersection_offroad + measurements.player_measurements.intersection_otherlane
        
        # print('speed = ' + str(speed) + 'collision = ' + str(collision) + 'intersection = ' + str(intersection))


        if reward is None:
            reward = (60 - abs(speed - 30) * 2.0 - collision / 50 - intersection * 100) / 100.0 #计算reward, speed距离30km/h的差，collision大概是在0~10000
            # reward = ( speed - collision / 50 - intersection * 100) / 100.0 #计算reward, speed距离30km/h
        return reward, [speed, collision, intersection]
        
    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None, None
        if keys[K_t]:
            self.should_display = not self.should_display
            return 'done', None
        if keys[K_m]:
            self.manual = True
            return 'done', None
        if keys[K_n]:
            self.manual = False
            return 'done', None
        if keys[K_v]:
            self.endnow = True
            return 'done', None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_c]:
            self.manual_control = not self.manual_control
        if keys[K_p]:#系统自带的自动驾驶
            self._enable_autopilot = not self._enable_autopilot
        control.reverse = self._is_on_reverse

        reward = None
        if keys[K_1]:
            reward = -1
        if keys[K_2]:
            reward = -0.5
        if keys[K_3]:
            reward = -0.25
        if keys[K_4]:
            reward = -0.1
        if keys[K_5]:
            reward = 0
        if keys[K_6]:
            reward = 0.1
        if keys[K_7]:
            reward = 0.25
        if keys[K_8]:
            reward = 0.5
        if keys[K_9]:
            reward = 1

        return control, reward

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation,
            reward):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road,'
        message += '{reward:.2f} reward.'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad,
            reward=reward)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
        if self.should_display == False:
            return
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            # print(array.shape)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        # if self._mini_view_image1 is not None:
        #     array = image_converter.depth_to_logarithmic_grayscale(self._mini_view_image1)
        #     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        #     self._display.blit(surface, (gap_x, mini_image_y))

        # if self._mini_view_image2 is not None:
        #     array = image_converter.labels_to_cityscapes_palette(
        #         self._mini_view_image2)
        #     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #     self._display.blit(
        #         surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

        # if self._lidar_measurement is not None:
        #     lidar_data = np.array(self._lidar_measurement.data[:, :2])
        #     lidar_data *= 2.0
        #     lidar_data += 100.0
        #     lidar_data = np.fabs(lidar_data)
        #     lidar_data = lidar_data.astype(np.int32)
        #     lidar_data = np.reshape(lidar_data, (-1, 2))
        #     #draw lidar
        #     lidar_img_size = (200, 200, 3)
        #     lidar_img = np.zeros(lidar_img_size)
        #     lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        #     surface = pygame.surfarray.make_surface(lidar_img)
        #     self._display.blit(surface, (10, 10))

        # if self._map_view is not None:
        #     array = self._map_view
        #     array = array[:, :, :3]

        #     new_window_width = \
        #         (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
        #         float(self._map_shape[1])
        #     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #     w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
        #     h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

        #     pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
        #     for agent in self._agent_positions:
        #         if agent.HasField('vehicle'):
        #             agent_position = self._map.convert_to_pixel([
        #                 agent.vehicle.transform.location.x,
        #                 agent.vehicle.transform.location.y,
        #                 agent.vehicle.transform.location.z])

        #             w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
        #             h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

        #             pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

        #     self._display.blit(surface, (WINDOW_WIDTH, 0))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            # print(np.array(array).shape)
            # print(self._map_shape)

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])

                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    wrapper = Carla_Wrapper()
    while True:
        try:

            with make_carla_client(args.host, args.port, timeout=1000) as client:
                game = CarlaGame(client, args, wrapper)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
