"""OpenAI gym environment for Carla. Run this file for a demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import atexit
import cv2
import os
import json
import random
import signal
import subprocess
import sys
import time
import traceback
import logging

import numpy as np
try:
	import scipy.misc
except Exception:
	pass

import gym
from gym.spaces import Box, Discrete, Tuple
from tmp_main import CarlaGame

from carla.client import VehicleControl, CarlaClient
from carla.tcp import TCPConnectionError


ENV_CONFIG = {
	"log_images": True,
	"enable_planner": True,
	"framestack": 2,  # note: only [1, 2] currently supported
	"convert_images_to_video": True,
	"early_terminate_on_collision": True,
	"verbose": True,
	"reward_function": "custom",
	"render_x_res": 800,
	"render_y_res": 600,
	"x_res": 80,
	"y_res": 80,
	"server_map": "/Game/Maps/Town02",
	"use_depth_camera": False,
	"discrete_actions": True,
	"squash_action_logits": False,
}


class CarlaEnv(gym.Env):

	def __init__(self, config=ENV_CONFIG):


		self.config = config
		self.city = self.config["server_map"].split("/")[-1]

		self.action_space = Discrete(13)

		image_space = Box(
			-1.0, 1.0, shape=(
				config["x_res"], config["y_res"],
				8), dtype=np.float32)

		self.observation_space = Tuple(
			[image_space, Box(-2.0, 2.0, shape=(1,), dtype=np.float32)])

	def reset(self):

		print('reset!')
		print('=.' * 40)

		a = 0
		log_level = logging.INFO
		logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
		
		if os.path.exists('port.txt'):
			with open('port.txt', 'r') as f:
				a = int(f.readlines()[0].strip())
				print(a)
				print('=.' * 40)

		with open('port.txt', 'w') as f:
			f.write(str(a + 1) + '\n')

		logging.info('listening to server %s:%s', 'localhost', 7899)

		while True:
			try:
				client = CarlaClient('localhost', 7899, timeout=1000)
				client.connect()
				self.carla_game = CarlaGame(client)
				self.carla_game.reset()
				obs = self.carla_game.get_obs()
				self.carla_game.display()
				return obs[0]

			except TCPConnectionError as error:
				logging.error(error)
				time.sleep(1)

	def step(self, action):

		print('step! ', action)
		print('=.' * 40)

		action = self.carla_game.do_action(action)
		obs = self.carla_game.get_obs()
		self.carla_game.display()
		
		return (obs[0], obs[1], obs[2], {})

