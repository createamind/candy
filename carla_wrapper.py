# -*- coding: utf-8 -*-

from candy_model import Machine
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

# BUFFER_LIMIT = 258
BATCH_SIZE = 128
KEEP_CNT = 1500
MAX_SAVE = 0
TRAIN_EPOCH = 15

class Carla_Wrapper(object):

	def __init__(self, gamma=0.99, lam=0.95):
		self.machine = Machine()
		self.global_step = 0

		self.lam = lam
		self.gamma = gamma
		self.state = self.machine.ppo.initial_state

		self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states, self.std_actions, self.manual = [],[],[],[],[],[],[],[],[]
		#obs:观察到的东西((连续的两帧包括深度的图像, speed), actions:采取的动作的code, values:PPO中critic的对当前局面的估价, neglogpacs：PPO中actor对当前动作的概率的负log, rewards：当前的奖励
		#vaerecons:VAE的重建误差，用于difficulty的计算,进行优先采样 states：ＰＰＯ中ＬＳＴＭ的中间状态, std_actions:人给的动作的ｃｏｄｅ, manual：是否应该使用人给的动作进行模仿学习
		self.last_frame = None
		# self.pretrain()


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

		t1 = np.array(image_converter.to_rgb_array(_main_image)).astype(np.float32) / 128 - 1
		t2 = np.max(image_converter.depth_to_logarithmic_grayscale(_mini_view_image1), axis=2, keepdims=True) / 128 - 1
		t3 = np.max(image_converter.to_rgb_array(_mini_view_image2), axis=2)
		return [t1, t2, t3]




	def update_reward(self, cnt, obs, action, reward):


		l = len(self.obs)
		for t in range(l - cnt, l - 2):
			self.rewards[t] = self.rewards[t+1]
		if reward is None:
			self.rewards[l-1] = self.rewards[l-2]
		else:
			self.rewards[l-1] = reward
		self.rewards[l-1] *= 20
		for t in reversed(range(l - cnt, l - 1)):
			self.rewards[t] += self.lam * self.rewards[t+1]




	def post_process(self, inputs, cnt):

		obs, reward, action, std_action, manual = self.pre_process(inputs)


		self.update_reward(cnt, obs, action, reward)



		print(self.rewards[-20:])
		print('Start Memory Replay')
		self.memory_training()
		print('Memory Replay Done')


	def pre_process(self, inputs, refresh=False):
		measurements, sensor_data, control, reward, collision, std_control, manual = inputs
		sensor_data = self.process_sensor_data(sensor_data)

		nowframe = np.concatenate([sensor_data[0], sensor_data[1]], 2)#深度和RGB图连接起来

		if self.last_frame is None:
			self.last_frame = nowframe

		frame = np.concatenate([self.last_frame, nowframe], 2)#连续两帧连续起来
		if refresh:
			self.last_frame = nowframe

		obs = (frame, measurements.player_measurements.forward_speed * 3.6 / 100) #obs分为当前frame和speed

		#将control从VehicleControl()变为数字
		action = self.analyze_control(control)
		std_action = self.analyze_control(std_control)
		if std_action == 0:
			manual = False
		return obs, reward, action, std_action, manual
		

	def update(self, inputs):

		obs, reward, action, std_action, manual = self.pre_process(inputs, refresh=True)




		self.states.append(self.state)

		_, value, self.state, neglogpacs, vaerecon = self.machine.value(obs, self.state, action)

		self.obs.append(obs)
		self.actions.append(action)
		self.values.append(value)
		self.neglogpacs.append(neglogpacs)
		self.rewards.append(reward)
		self.vaerecons.append(vaerecon)
		self.std_actions.append(std_action)
		self.manual.append(manual)

		# self.red_buffer.append(red)
		# self.manual_buffer.append(manual)

	def pretrain(self):
		raise NotImplementedError
		# if os.path.exists('obs/data'):
		# 	print('Start Pretraining!!')
		# 	with open('obs/data', 'rb') as fp:
		# 		self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states = msgpack.load(fp, encoding='utf-8', raw=False)
		# 	print('Pretraining length = ', len(self.obs))
		# 	self.memory_training(pretrain=True)

	def calculate_difficulty(self, reward, vaerecon):
		# return abs(reward)
		return 1
		
	def memory_training(self, pretrain=False):
		l = len(self.obs)
		batch = []
		difficulty = []
		for i in range(l):
			batch.append([self.obs[i], self.actions[i], self.values[i], self.neglogpacs[i], self.rewards[i], self.vaerecons[i], self.states[i], self.std_actions[i], self.manual[i]])
			difficulty.append(self.calculate_difficulty(self.rewards[i], self.vaerecons[i]))
		# print(self.rewards)
		# print(self.values)
		# print(np.array(self.rewards) - np.array([i[0] for i in self.values]))
		difficulty = np.array(difficulty)
		print(difficulty[-20:])
		def softmax(x):
			x = np.clip(x, 1e-3, 1)
			return np.exp(x) / np.sum(np.exp(x), axis=0)
		difficulty = softmax(difficulty * 5)
		print(difficulty[-20:])
		print("Memory Extraction Done.")

		for _ in tqdm(range(TRAIN_EPOCH)):
			roll = np.random.choice(len(difficulty), BATCH_SIZE, p=difficulty)
			tbatch = []
			for i in roll:
				tbatch.append(batch[i])
			tra_batch = [np.array([t[i] for t in tbatch]) for i in range(9)]
			# tra_batch = [np.array([t[i] for t in tbatch]) for i in range(7)]
			self.machine.train(tra_batch, self.global_step)
			self.global_step += 1

		self.machine.save()

		# if pretrain:          
		# 	self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states = [],[],[],[],[],[],[]

		if len(self.obs) > KEEP_CNT:
			rem = len(self.obs) - KEEP_CNT
			self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states, self.std_actions, self.manual = \
				self.obs[rem:],self.actions[rem:],self.values[rem:],self.neglogpacs[rem:],self.rewards[rem:],self.vaerecons[rem:], self.states[rem:], self.std_actions[rem:], self.manual[rem:]

	def decode_control(self, cod):
		#将数字的ｃｏｎｔｒｏｌ转换为VehicleControl()
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
			elif cod == 11:
				control.steer = 1
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

	def get_control(self, inputs):
		obs, reward, action, std_action, manual = self.pre_process(inputs)
		action, _, _, _, _ = self.machine.step(obs, self.state)#整个模型跑一步
		control = self.decode_control(action)
		return control
