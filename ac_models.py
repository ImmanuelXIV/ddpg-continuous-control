#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
	""" Hidden layer initialization, based on https://arxiv.org/abs/1509.02971."""
	fan_in = layer.weight.data.size()[0]
	lim = 1. / np.sqrt(fan_in)
	return (-lim, lim)


class Actor(nn.Module):
	""" Policy network that maps states to action probs."""

	def __init__(self, state_size, action_size, seed):
		""" Initialize neural network. 
		Params
		======
			state_size:  	dimension of each state
			action_size: 	dimension of each action
			seed:			seed
		"""
		super().__init__()
		torch.manual_seed(seed)

		self.fc1 = nn.Linear(state_size, 200)
		self.bn1 = nn.BatchNorm1d(num_features=200)
		self.fc2 = nn.Linear(200, 300)
		self.bn2 = nn.BatchNorm1d(num_features=300)
		self.fc3 = nn.Linear(300, action_size)
		self.reset_parameters()


	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)


	def forward(self, state):
		x = F.relu(self.bn1(self.fc1(state)))
		x = F.relu(self.bn2(self.fc2(x)))
		return torch.tanh(self.fc3(x))


class Critic(nn.Module):
	""" Value network that maps states and actions to its action value."""

	def __init__(self, state_size, action_size, seed):
		""" Initialize neural network. 
		Params
		======
			state_size:  	dimension of each state
			action_size: 	dimension of each action
			seed:			seed
		"""
		super().__init__()
		torch.manual_seed(seed)

		self.fc1 = nn.Linear(state_size, 200)
		self.bn1 = nn.BatchNorm1d(num_features=200)
		self.fc2 = nn.Linear(200 + action_size, 300)
		self.bn2 = nn.BatchNorm1d(num_features=300)
		self.fc3 = nn.Linear(300, 1)
		self.reset_parameters()


	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)


	def forward(self, state, action):
		x = F.relu(self.bn1(self.fc1(state)))
		x = torch.cat((x, action), dim=1)
		x = F.relu(self.bn2(self.fc2(x)))
		return self.fc3(x)

