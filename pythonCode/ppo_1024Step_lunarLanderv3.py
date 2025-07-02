import gymnasium as gym 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical



env = gym.make("LunarLander-v3", render_mode="human")

print("Continuous Obsevation space shape: ", env.observation_space.shape[0])
print("Observation space, Lower bounds: ", env.observation_space.low)
print("Observation space, Upper bounds: ", env.observation_space.high)

print("Discrete Action space .n: ", env.action_space.n)
print("Action values: ", list(range(env.action_space.n)))

action_descriptions = {
0: "Do nothing", 
1: "Fire left engine", 
2: "Fire main engine", 
3: "Fire right engine" } 

print("\n Available actions:")
for action in range(env.action_space.n):
	print(f" {action}: {action_descriptions[action]}")



#hyperparameters 
learning_rate = 1e-3
gamma = 0.99
lam = 0.95
num_episodes = 3010
rollout_length = 4*1024
eps_clip = 0.2
num_epochs = 10



class PPO_Actor(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		
		#dim=0 yields normalizing down the column
		#dim=1 yields normalizing across the row
		#dim=-1 yields normalizing the last dimension
		self.net = nn.Sequential( 
			nn.Linear(input_dim, 128),
			nn.ReLU(),
			nn.Linear(128, output_dim),
			nn.Softmax(dim=-1)
		)

	def forward(self, x):
		return self.net(x)

class PPO_Critic(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		
		self.net = nn.Sequential(
			nn.Linear(input_dim, 128), 
			nn.ReLU(),
			nn.Linear(128, 1)
		)

	#squeeze() removes all dimensions of size 1
	#squeeze(-1) removes the last dimension if it is size 1
	#squeeze(0) removes the first dimension if it is size 1
	def forward(self, x):
		return self.net(x).squeeze(-1)



def compute_GAE(next_value, rewards, values, dones, gamma, lam):
	advantages = []
	returns = []
	GAE = 0
	G = 0

	values = values + [next_value]

	for t in reversed(range(len(rewards))):
		deltaGAE = rewards[t] + gamma * values[t+1] - values[t]
		GAE = deltaGAE + gamma * lam * GAE
		advantages.insert(0, GAE)
	
	for reward in reversed(rewards):
		G = reward + gamma * G
		returns.insert(0,G)

	return returns, advantages

def compute_returns(rewards, dones, gamma):
	returns = []
	G = torch.tensor(0.0)
	#zip makes it a pair	
	for reward, done in zip(reversed(rewards), reversed(dones)):
		G = reward + gamma * G * (1 - done) 
		returns.insert(0,G)
	return returns



#training loop
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

ppoActor = PPO_Actor(observation_dim, action_dim)
optimPPOActor = optim.Adam(ppoActor.parameters(), lr=learning_rate)

ppoCritic = PPO_Critic(observation_dim)
optimPPOCritic = optim.Adam(ppoCritic.parameters(), lr=learning_rate) 

resume_training = True #False
if resume_training:
	checkpoint = torch.load("ppo_1024Step_checkpoint.pth")
	ppoActor.load_state_dict(checkpoint["actor_state_dict"])
	ppoCritic.load_state_dict(checkpoint["critic_state_dict"])
	optimPPOCritic.load_state_dict(checkpoint["optimizer_critic_state_dict"])
	start_episode = checkpoint["episode"] + 1
	print(f"Resuming training from episode {start_episode}")
else: 
	start_episode = 0

for episode in range(start_episode, num_episodes):
	#oldPPOActor = PPO_Actor(observation_dim, action_dim)
	#oldPPOActor.load_state_dict(ppoActor.state_dict()) 
	
	obs = []
	actions = [] 	
	rewards = []
	dones = []
	log_probs_old = []
	values = []

	ob, _ = env.reset()
	ob = torch.tensor(ob, dtype=torch.float32)

	for _ in range(rollout_length): 
		#action is not a pytorch tensor, but just an index
		#action = env.action_space.sample()
	
		with torch.no_grad():	
			#probs is 1D torch.tensor with n elements 
			probs = ppoActor(ob) #oldPPOActor(ob)
			#dist is not a tensor, but a distribution object
			dist = Categorical(probs)
			#action is a pytorch tensor with value 0, 1, 2, ...
			action = dist.sample()		
			#log_prob is log of probability associated with selected softmax action / item 
			#log_prob is a torch scalar	
			log_prob_old = dist.log_prob(action)
			value = ppoCritic(ob)

		#ob_next is numpy array => torch.tensor(ob_next)
		#reward is scalar => torch.tensor(reward), list of scalars => torch.tensor(rewards)
		ob_next, reward, terminated, truncated, info = env.step(action.item())
		ob_next = torch.tensor(ob_next, dtype=torch.float32)
		done = terminated or truncated		

		#store everything
		obs.append(ob)
		actions.append(action)
		rewards.append(reward)
		dones.append(done)
		log_probs_old.append(log_prob_old)
		values.append(value)
	
		ob = ob_next

		if done: 
			ob, _ = env.reset()
			ob = torch.tensor(ob, dtype=torch.float32)

	with torch.no_grad():
		next_value = ppoCritic(ob)

	returns, advantages = compute_GAE(next_value, rewards, values, dones, gamma, lam)

	obs = torch.stack(obs)
	actions = torch.stack(actions)
	log_probs_old = torch.stack(log_probs_old)
	returns = torch.tensor(returns, dtype=torch.float32) 
	advantages = torch.tensor(advantages, dtype=torch.float32)
	values = torch.stack(values) 

	#normalize
	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

	checkpoint = {
                "episode": episode,
                "actor_state_dict": ppoActor.state_dict(),
                "critic_state_dict": ppoCritic.state_dict(),
                "optimizer_actor_state_dict": optimPPOActor.state_dict(),
                "optimizer_critic_state_dict": optimPPOCritic.state_dict()
        }


	torch.autograd.set_detect_anomaly(True)

	for _ in range(num_epochs):
		valuesR = ppoCritic(obs)
	
		probs = ppoActor(obs)
		dist = Categorical(probs)
	
		log_probs_new = dist.log_prob(actions)

		ratios = (log_probs_new - log_probs_old).exp()

		surrogateNoClip = ratios * advantages
		surrogateClip = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

		#print("new log probs shape:", log_probs_new.shape)
		#print("ratios shape:", ratios.shape)
		#print("advantages shape:", advantages.shape)
		#print("returns shape:", returns.shape)
		#print("values shape:", values.shape)

		actorLoss2 = -torch.min(surrogateNoClip, surrogateClip)
		criticLoss2 = F.mse_loss(valuesR, returns)
		
		#use stack so computational graph is not broken
		ppoActor_loss = actorLoss2.mean()

		optimPPOActor.zero_grad()
		ppoActor_loss.backward()
		optimPPOActor.step()

		ppoCritic_loss = criticLoss2

		optimPPOCritic.zero_grad()
		ppoCritic_loss.backward()
		optimPPOCritic.step()

	print(f"Episode {episode}, actorLoss {ppoActor_loss:.5f}, criticLoss {ppoCritic_loss:.2f}")


torch.save(checkpoint, "ppo_1024Step_checkpoint.pth")
print("Checkpoint saved.")

env.close()
