#!/usr/bin/python
import random
import numpy as np
import torch
from collections import deque, namedtuple



class ReplayBuffer():
	"""
	Replay buffer to store and sample experience tuples.
	"""
	
	def __init__(self, buffer_size, batch_size, device, seed):
		""" Initialize replay buffer.

		Params
		======
			buffer_size:	size of replay buffer
			batch_size:  	batch size for training
			device:      	gpu/cpu
			seed:		 	seed
		"""
		self.buffer_size = buffer_size
		self.buffer = deque(maxlen=self.buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
		self.device = device
		random.seed(seed)


	def add(self, states, actions, rewards, next_states, dones):
		for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
			# Multiple agents collect experience, but we add (s, a, r, ns, d) tuples separately to the buffer. 
			e = self.experience(state, action, reward, next_state, done)
			self.buffer.append(e)


	def sample(self):
		experience = random.sample(self.buffer, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experience if not None])).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experience if not None])).float().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experience if not None])).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if not None])).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experience if not None]).astype(np.uint8)).float().to(self.device)
		return (states, actions, rewards, next_states, dones)


	def reset(self):
		self.buffer = deque(maxlen=self.buffer_size)


	def __len__(self):
		return len(self.buffer)
