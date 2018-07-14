#!/usr/bin/env python3

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
    n            : set auto driving
    m            : set manual pilot
    t            : toggle display
    v            : save model(in the next memory replay stage)

STARTING in a moment...
"""

from __future__ import print_function

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

STEP_LIMIT = 200
MAP_NAME = 'Town01'




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
    def __init__(self, carla_client):
        self.client = carla_client
        self._carla_settings = self.make_carla_settings()
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = True
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = MAP_NAME
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        self.for_save = False
        self.should_display = True
        random.seed(datetime.datetime.now())
        self.manual = (random.randint(1,2) != 1)
        self.step = 0
        self.history_collision = 0

        self.prev_image = None
        

    
    def make_carla_settings(self):
        """Make a CarlaSettings object with the settings we need."""
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=15,
            NumberOfPedestrians=30,
            WeatherId=random.choice([1, 3, 7, 8, 14]),
            QualityLevel='Epic')

        settings.randomize_seeds()
        camera0 = sensor.Camera('CameraRGB')
        camera0.set_image_size(80, 80)
        camera0.set_position(2.0, 0.0, 1.4)
        camera0.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera0)
        camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
        camera1.set_image_size(80, 80)
        camera1.set_position(2.0, 0.0, 1.4)
        camera1.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera1)
        camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
        camera2.set_image_size(80, 80)
        camera2.set_position(2.0, 0.0, 1.4)
        camera2.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera2)

        camera3 = sensor.Camera('CameraForHuman')
        camera3.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        camera3.set_position(2.0, 0.0, 1.4)
        camera3.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera3)

        # if args.lidar:
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

        
    def reset(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()

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
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def get_obs(self):

        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        self.auto_control = measurements.player_measurements.autopilot_control

        self._main_image = sensor_data.get('CameraForHuman', None)


        rgb_image = sensor_data.get('CameraRGB', None)
        depth_image = sensor_data.get('CameraDepth', None)


        rgb_image = image_converter.to_rgb_array(rgb_image)
        rgb_image = (rgb_image.astype(np.float32) - 128) / 128
        depth_image = (np.max(image_converter.depth_to_logarithmic_grayscale(depth_image), axis=2, keepdims=True) - 128) / 128
        image = np.concatenate([rgb_image, depth_image], axis=2)


        if self.prev_image is None:
            self.prev_image = image

        now_image = image
        image = np.concatenate([self.prev_image, image], axis=2)

        self.prev_image = now_image

        collision = self.get_collision(measurements)

        control, reward = self._get_keyboard_control(pygame.key.get_pressed())
        reward, _ = self.calculate_reward(measurements, reward, collision)


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
        self.step += 1
        done = False
        if self.step > STEP_LIMIT:
            done = True
            self.step = 0
            
        return ((image, [measurements.player_measurements.forward_speed * 3.6 / 100]), reward, done)

    def do_action(self, action):

        print('=.' * 40)
        print('do action!', action)

        control, reward = self._get_keyboard_control(pygame.key.get_pressed())
        if control == "done":
            control = self.auto_control
            self.client.send_control(control)
            return
        elif control is None:
            self._on_new_episode()
            return

        if self.manual:
            if self._enable_autopilot:
                control = self.auto_control
        else:
            control = self.decode_control(action)

        print(control)
        self.client.send_control(control)
        return self.analyze_control(control)

    def get_collision(self, measurements):
        new_collision = measurements.player_measurements.collision_vehicles + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_other
        ans = new_collision - self.history_collision
        self.history_collision = new_collision
        return ans

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

    def calculate_reward(self, measurements, reward, collision):
        
        speed = abs(measurements.player_measurements.forward_speed) * 3.6

        # collision = measurements.player_measurements.collision_vehicles + measurements.player_measurements.collision_pedestrians + measurements.player_measurements.collision_other

        # intersection = measurements.player_measurements.intersection_otherlane + measurements.player_measurements.intersection_offroad
        
        intersection = measurements.player_measurements.intersection_offroad
        
        # print('speed = ' + str(speed) + 'collision = ' + str(collision) + 'intersection = ' + str(intersection))

        if reward is None:
            reward = (speed - collision / 50 - intersection * 100) / 100.0

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
            self.for_save = True
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
        if keys[K_p]:
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

    def display(self):
        if self.should_display == False:
            return
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
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

        if self._lidar_measurement is not None:
            lidar_data = np.array(self._lidar_measurement.data[:, :2])
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            #draw lidar
            lidar_img_size = (200, 200, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            surface = pygame.surfarray.make_surface(lidar_img)
            self._display.blit(surface, (10, 10))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

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
