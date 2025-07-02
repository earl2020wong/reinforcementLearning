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
num_episodes = 2000 



class ReinforcePolicy(nn.Module):
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

reinforcePolicy = ReinforcePolicy(observation_dim, action_dim)
optimizer = optim.Adam(reinforcePolicy.parameters(), lr=learning_rate)



resume_training = False
if resume_training:
	checkpoint = torch.load("reinforce_checkpoint.pth")
	reinforcePolicy.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	start_episode = checkpoint["episode"] + 1
	print(f"Resuming training from episode {start_episode}")
else: 
	start_episode = 0

for episode in range(start_episode, num_episodes):
	log_probs = []
	rewards = []
	actions = []

	obs, _ = env.reset()
	done = False
	total_reward = 0.0

	while not done: 
		#action = env.action_space.sample()
		
		obs_tensor = torch.tensor(obs, dtype=torch.float32)
		probs = reinforcePolicy(obs_tensor)
		dist = torch.distributions.Categorical(probs)
		action = dist.sample()		

		log_probs.append(dist.log_prob(action))
	
		obs, reward, terminated, truncated, info = env.step(action.item())
		done = terminated or truncated 
		total_reward += reward

		#time.sleep(0.05) 

		rewards.append(reward)
		actions.append(action)

	returns = compute_returns(rewards, gamma)
	returns = torch.tensor(returns, dtype=torch.float32)
	returns = (returns - returns.mean()) / (returns.std() + 1e-8)

	rPL = []
	#rPL = [-log_prob * G for log_prob, G in zip(log_probs, returns)]

	for i in range(len(log_probs)):
		log_prob = log_probs[i]
		G = returns[i]
		loss = -log_prob * G
		rPL.append(loss)

	reinforcePolicy_loss = torch.stack(rPL).sum()

	optimizer.zero_grad()
	reinforcePolicy_loss.backward()
	optimizer.step()

	print(f"Episode {episode}, reward_received {reward:.2f}, total reward {total_reward:.2f}")

checkpoint = {
    "episode": episode,
    "model_state_dict": reinforcePolicy.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}

torch.save(checkpoint, "reinforce_checkpoint.pth")
print("Checkpoint saved.")

env.close()
