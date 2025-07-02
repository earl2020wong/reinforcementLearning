import gymnasium as gym 
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



env = gym.make("LunarLander-v3") #, render_mode="human")

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
num_episodes = 3010 



class A2C_Actor(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()

		self.net = nn.Sequential( 
			nn.Linear(input_dim, 128),
			nn.ReLU(),
			nn.Linear(128, output_dim),
			nn.Softmax(dim=-1)
		)

	def forward(self, x):
		return self.net(x)

class A2C_Critic(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		
		self.net = nn.Sequential(
			nn.Linear(input_dim, 128), 
			nn.ReLU(),
			nn.Linear(128, 1)
		)

	def forward(self, x):
		return self.net(x).squeeze(-1)



def compute_returns(rewards, gamma):
	returns = []
	G = 0
	for r in reversed(rewards):
		G = r + gamma * G
		returns.insert(0,G)
	return returns



#training loop
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

a2cActor = A2C_Actor(observation_dim, action_dim)
optimA2CActor = optim.Adam(a2cActor.parameters(), lr=learning_rate)

a2cCritic = A2C_Critic(observation_dim)
optimA2CCritic = optim.Adam(a2cCritic.parameters(), lr=learning_rate) 



resume_training = False #True #False
if resume_training:
	checkpoint = torch.load("a2c_td0_checkpoint.pth")
	a2cActor.load_state_dict(checkpoint["actor_state_dict"])
	optimA2CActor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
	a2cCritic.load_state_dict(checkpoint["critic_state_dict"])
	optimA2CCritic.load_state_dict(checkpoint["optimizer_critic_state_dict"])
	start_episode = checkpoint["episode"] + 1
	print(f"Resuming training from episode {start_episode}")
else: 
	start_episode = 0

for episode in range(start_episode, num_episodes):
	log_probs = []
	rewards = []
	actions = []
	advantageLoss = []
	criticLoss = []

	obs, _ = env.reset()
	obs = torch.tensor(obs, dtype=torch.float32)

	done = False
	total_reward = 0.0

	while not done: 
		#action = env.action_space.sample()		
		#obs_tensor = torch.tensor(obs, dtype=torch.float32)

		probs = a2cActor(obs)
		dist = torch.distributions.Categorical(probs)
		action = dist.sample()		

		log_probs.append(dist.log_prob(action))
	
		obs_next, reward, terminated, truncated, info = env.step(action.item())
		obs_next = torch.tensor(obs_next, dtype=torch.float32)
		done = terminated or truncated 
		total_reward += reward

		#actor
		rewards.append(reward)
		actions.append(action)

		#critic
		#with torch.no_grad():
		value_next = a2cCritic(obs_next)
		value = a2cCritic(obs)

		target = reward + (1 - done) * gamma * value_next
		advantageLoss.append(target - value)
		#detach() prevents gradients from propagating through 
		critic_loss = (target - value).pow(2)
		criticLoss.append(critic_loss)

		obs = obs_next

	#returns = compute_returns(rewards, gamma)
	#returns = torch.tensor(returns, dtype=torch.float32)
	#returns = (returns - returns.mean()) / (returns.std() + 1e-8)

	
	actorLoss = []
	#rPL = [-log_prob * G for log_prob, G in zip(log_probs, returns)]

	for i in range(len(log_probs)):
		log_prob = log_probs[i]
		G = torch.tensor(advantageLoss[i], dtype=torch.float32) #returns[i]
		actor_loss = -log_prob * G
		actorLoss.append(actor_loss)

	a2cActor_loss = torch.stack(actorLoss, dim=0).mean()

	optimA2CActor.zero_grad()
	a2cActor_loss.backward()
	optimA2CActor.step()

	#criticLoss = torch.tensor(criticLoss, dtype=torch.float32)
	#use stack so computational graph is not broken 
	a2cCritic_loss = torch.stack(criticLoss).mean()

	optimA2CCritic.zero_grad()
	a2cCritic_loss.backward()
	optimA2CCritic.step()

	print(f"Episode {episode}, reward_received {reward:.2f}, total reward {total_reward:.2f} criticLoss {a2cCritic_loss:.2f}")

checkpoint = {
    "episode": episode,
    "actor_state_dict": a2cActor.state_dict(),
    "critic_state_dict": a2cCritic.state_dict(), 
    "optimizer_actor_state_dict": optimA2CActor.state_dict(), 
    "optimizer_critic_state_dict": optimA2CCritic.state_dict()
}

torch.save(checkpoint, "a2c_td0_checkpoint.pth")
print("Checkpoint saved.")

env.close()
