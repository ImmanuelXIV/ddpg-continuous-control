#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ac_models import Actor, Critic
from noise import OUnoise
from replaybuffer import ReplayBuffer



SEED = 123				# random seed
GAMMA = 0.98			# discounting factor
LR_A = 1e-4				# learning rate actor
LR_C = 1e-3				# learning rate actor
WEIGEHT_DECAY = 0		# weight decay
TAU = 1e-3				# soft update parameter
BUFFER_SIZE = int(1e6)	# replay buffer size
BATCH_SIZE = 128		# batch size
C = 20					# update the nets every C time steps
D = 10					# update the nets D times

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent():
	""" Actor critic agent that implements the DDPG algorithm."""

	def __init__(self, state_size, action_size):
		""" Initialize the agent.
		Params
		======
			state_size:		dimension of each state
			action_size: 	dimension of each action
		"""
		self.state_size = state_size
		self.action_size = action_size
		self.t_step = 0

		self.actor_local = Actor(self.state_size, self.action_size, SEED).to(DEVICE)
		self.actor_target = Actor(self.state_size, self.action_size, SEED).to(DEVICE)
		self.actor_target.load_state_dict(self.actor_local.state_dict())
		self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_A)

		self.critic_local = Critic(self.state_size, self.action_size, SEED).to(DEVICE)
		self.critic_target = Critic(self.state_size, self.action_size, SEED).to(DEVICE)
		self.critic_target.load_state_dict(self.critic_local.state_dict())
		self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_C, weight_decay=WEIGEHT_DECAY)

		self.noise = OUnoise(self.action_size, SEED)
		self.replaybuffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, DEVICE, SEED)


	def select_action(self, states, ou_noise=True):
		""" Return action probs, add noise for random exploration in training mode.
		Params
		======
			states:		states of shape [num_agents, state_dim]
			ou_noise: 	add noise for exploration in training mode
		"""
		states = torch.from_numpy(states).float().to(DEVICE) # torch.Size([20, 33])
		self.actor_local.eval()

		with torch.no_grad():
			actions = self.actor_local.forward(states).cpu().data.numpy() # (20, 4)

		if ou_noise:
			actions += self.noise.sample_noise()

		self.actor_local.train()
		return actions


	def step(self, states, actions, rewards, next_states, dones):
		""" Add experience to replay buffer and make learning step."""
		self.replaybuffer.add(states, actions, rewards, next_states, dones)
		
		self.t_step = (self.t_step + 1) % C
		if self.t_step == 0: # update every C time steps
			if len(self.replaybuffer) >= BATCH_SIZE:
				self.learn()


	def learn(self): 
		""" Sample batches, update actor/critic (local & target) nets, see https://arxiv.org/abs/1509.02971."""
		for _ in range(D):
			# sample experience
			states, actions, rewards, next_states, dones = self.replaybuffer.sample()

			# ------------------------- Critic update ------------------------- #
			next_actions = self.actor_target(next_states)
			Qtargets_next = self.critic_target(next_states, next_actions)
			Qtargets = rewards + (GAMMA * Qtargets_next * (1 - dones))
			Qexpected = self.critic_local(states, actions)
			critic_loss = F.mse_loss(Qtargets, Qexpected)
			
			self.critic_optim.zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
			self.critic_optim.step()

			# ------------------------- Actor update ------------------------- #
			actions_pred = self.actor_local(states)
			actor_loss = -self.critic_local(states, actions_pred).mean()

			self.actor_optim.zero_grad()
			actor_loss.backward()
			self.actor_optim.step()

			# ------------------------- Soft update ------------------------- #
			self.soft_update(self.critic_target, self.critic_local)
			self.soft_update(self.actor_target, self.actor_local)


	def soft_update(self, target, local):
		""" Copy local params into target params via Tau * local.params + (1-Tau) * target.params."""
		for target_params, local_params in zip(target.parameters(), local.parameters()):
				target_params.data.copy_(TAU * local_params.data + (1 - TAU) * target_params.data)


