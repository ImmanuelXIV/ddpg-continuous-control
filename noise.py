#!/usr/bin/python
import random
import numpy as np
import copy


class OUnoise():
	"""
	Ornstein–Uhlenbeck process noise, see https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
	"""
	
	def __init__(self, action_size, seed, mu=0.0, theta=0.15, sigma=0.2):
		""" Initialize OU noise. 
		Params
		======
			action_size: 	dimension of each action
			seed:			seed
			mu:				mean
			theta:			mean reversion rate
			sigma:			std
		"""
		self.mu = np.ones(action_size) * mu
		self.theta = theta
		self.sigma = sigma
		random.seed(seed)
		self.reset()

        
	def sample_noise(self):
		ns = self.state
		dn = self.theta * (self.mu - ns) + self.sigma * np.array([np.random.randn() for i in range(len(ns))])
		self.state = ns + dn
		return self.state

    
	def reset(self):
		self.state = copy.copy(self.mu)
