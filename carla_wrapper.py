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
TRAIN_EPOCH = 30

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

		# b = 0
		# if steer < -0.2:
		# 	b += 2
		# elif steer > 0.2:
		# 	b += 1
		# b *= 3

		# if throttle > 0.4:
		# 	b += 2
		# if brake > 0.4:
		# 	b += 1

		# if reverse:
		# 	b = 9
		
		# if hand_brake:
		# 	if steer < -0.4:
		# 		b = 10
		# 	elif steer > 0.4:
		# 		b = 11
		# 	else:
		# 		b = 12


		th = 0
		if brake > 0:
			th = -brake
		if throttle > 0:
			th = throttle

		return [th, steer]

	def process_sensor_data(self, sensor_data):
		_main_image = sensor_data.get('CameraRGB', None)
		_mini_view_image1 = sensor_data.get('CameraDepth', None)
		_mini_view_image2 = sensor_data.get('CameraSemSeg', None)

		t1 = np.array(image_converter.to_rgb_array(_main_image)).astype(np.float32) / 128 - 1
		t2 = np.max(image_converter.depth_to_logarithmic_grayscale(_mini_view_image1), axis=2, keepdims=True) / 128 - 1
		t3 = np.max(image_converter.to_rgb_array(_mini_view_image2), axis=2)
		return [t1, t2, t3]


	# def update_reward(self, cnt, obs, action):
	# 	_, last_values, _, _, _ = self.machine.value(obs, self.state, action)

	# 	#discount/bootstrap off value fn
	# 	self.advs = np.zeros_like(self.rewards)
	# 	# self.rewards[-1] = reward
	# 	# print(' '.join([('%.2f' % i)for i in self.rewards]))
	# 	l = len(self.obs)
	# 	lastgaelam = 0
	# 	for t in reversed(range(l - cnt, l)):
	# 		if t == l - 1:
	# 			nextvalues = last_values
	# 		else:
	# 			nextvalues = self.values[t+1]
	# 		delta = self.rewards[t] + self.gamma * nextvalues - self.values[t]
	# 		self.advs[t] = lastgaelam = delta + self.gamma * self.lam * lastgaelam
	# 	for t in range(l - cnt, l):
	# 		self.rewards[t] = float(self.advs[t] + self.values[t])

	def update_reward(self, cnt, obs, action, reward):

		# #discount/bootstrap off value fn
		# self.advs = np.zeros_like(self.rewards)
		# # self.rewards[-1] = reward
		# # print(' '.join([('%.2f' % i)for i in self.rewards]))
		# lastgaelam = 0
		# for t in reversed(range(l - cnt, l)):
		# 	if t == l - 1:
		# 		nextvalues = last_values
		# 	else:
		# 		nextvalues = self.values[t+1]
		# 	delta = self.rewards[t] + self.gamma * nextvalues - self.values[t]
		# 	self.advs[t] = lastgaelam = delta + self.gamma * self.lam * lastgaelam
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

		#batch of steps to batch of rollouts
		# self.obs = np.asarray(self.obs, dtype=np.float32)
		# self.actions = np.asarray(self.actions, dtype=np.int32)
		# self.values = np.asarray(self.values, dtype=np.float32)
		# self.neglogpacs = np.asarray(self.neglogpacs, dtype=np.float32)
		# self.rewards = np.asarray(self.rewards, dtype=np.float32)
		# self.vaerecons = np.asarray(self.vaerecons, dtype=np.float32)
		# self.states = np.asarray(self.states, dtype=np.float32)

		self.update_reward(cnt, obs, action, reward)

		# print(' '.join([('%.2f' % i)for i in self.rewards]))

		# self.rewards = np.array(self.rewards)
		# if os.path.exists('obs/data'):
		# 	with open('obs/data', 'rb') as fp:
		# 		obs, actions, values, neglogpacs, rewards, vaerecons, states = msgpack.load(fp, encoding='utf-8', raw=False)
		# else:
		# 	obs, actions, values, neglogpacs, rewards, vaerecons, states = [], [], [], [], [], [], []
		
		# if len(obs) < MAX_SAVE:
		# 	with open('obs/data', 'wb') as fp:
		# 		msgpack.dump([obs + self.obs[l - BUFFER_LIMIT:l],\
		# 		 actions + self.actions[l - BUFFER_LIMIT:l],\
		# 		 values + self.values[l - BUFFER_LIMIT:l],\
		# 		 neglogpacs + self.neglogpacs[l - BUFFER_LIMIT:l],\
		# 		 rewards + self.rewards[l - BUFFER_LIMIT:l],\
		# 		 vaerecons + self.vaerecons[l - BUFFER_LIMIT:l],\
		# 		 states + self.states[l - BUFFER_LIMIT:l]],\
		# 		 fp, use_bin_type=True)
			

		print(self.rewards[-20:])
		print('Start Memory Replay')
		self.memory_training()
		print('Memory Replay Done')


	def pre_process(self, inputs, refresh=False):
		measurements, sensor_data, control, reward, steer, std_control, manual = inputs
		sensor_data = self.process_sensor_data(sensor_data)

		nowframe = np.concatenate([sensor_data[0], sensor_data[1]], 2)#深度和RGB图连接起来

		if self.last_frame is None:
			self.last_frame = nowframe

		frame = np.concatenate([self.last_frame, nowframe], 2)#连续两帧连续起来
		if refresh:
			self.last_frame = nowframe

		obs = (frame, measurements.player_measurements.forward_speed * 3.6 / 15.0 - 1, steer) #obs分为当前frame和speed

		#将control从VehicleControl()变为数字
		action = self.analyze_control(control)
		std_action = self.analyze_control(std_control)
		# if std_action == 0:
		# 	manual = False
		return obs, reward, action, std_action, manual
		

	def update(self, inputs):

		obs, reward, action, std_action, manual = self.pre_process(inputs, refresh=True)

		# sensor_data = self.process_sensor_data(sensor_data)
		# obs = [sensor_data[0]]
		# auxs = [sensor_data[1], sensor_data[2],\
		#     abs(measurements.player_measurements.forward_speed) * 3.6 / 100, \
		#     collision / 20000,\
		#     measurements.player_measurements.intersection_offroad]

		# control = self.analyze_control(control)
		# reward = [reward]
		
		# assert len(self.obs_buffer) == len(self.auxs_buffer) and len(self.auxs_buffer) == len(self.control_buffer)\
		#      and len(self.control_buffer) == len(self.reward_buffer)

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

		th, steer = cod
		if th > 0:
			control.throttle = min(th, 1.0)
		if th < 0:
			control.brake = min(-th, 1.0)

		control.steer = max(min(steer, 1.0), -1.0)

		# if cod > 9:
		# 	control.hand_brake = True
		# 	if cod == 10:
		# 		control.steer = -1
		# 	elif cod == 11:
		# 		control.steer = 1
		# 	return control

		# if cod == 9:
		# 	control.reverse = True
		# 	control.throttle = 1
		# 	return control

		# if cod % 3 == 1:
		# 	control.brake = 1
		# elif cod % 3 == 2:
		# 	control.throttle = 1
		
		# if cod // 3 == 1:
		# 	control.steer = 1
		# elif cod // 3 == 2:
		# 	control.steer = -1
		
		return control

	def get_control(self, inputs):
		obs, reward, action, std_action, manual = self.pre_process(inputs)
		print(action)
		print(std_action)
		action, _, _, _, _ = self.machine.step(obs, self.state)#整个模型跑一步
		print(action)
		control = self.decode_control(action[0])
		return control
