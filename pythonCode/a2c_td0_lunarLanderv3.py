#gymnasium is a community maintained fork of openAI gym.  
import gymnasium as gym 
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



env = gym.make("LunarLander-v3", render_mode="human")

print("Continuous Observation space shape: ", env.observation_space.shape[0])
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
num_episodes = 2015  



class A2C_Actor(nn.Module):
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

class A2C_Critic(nn.Module):
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



#training loop
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

a2cActor = A2C_Actor(observation_dim, action_dim)
optimA2CActor = optim.Adam(a2cActor.parameters(), lr=learning_rate)

a2cCritic = A2C_Critic(observation_dim)
optimA2CCritic = optim.Adam(a2cCritic.parameters(), lr=learning_rate) 



resume_training = True #False
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
	advantages = []
	actor_list = []
	critic_list = []

	#obs is a numpy array => torch.tensor(obs)
	obs, _ = env.reset()
	obs_tensor = torch.tensor(obs, dtype=torch.float32)

	done = False
	total_reward = 0.0

	while not done: 
		#action is not a pytorch tensor, but just an index
		#action = env.action_space.sample()	
	
		#probs is 1D torch.tensor with n elements 
		probs = a2cActor(obs_tensor)
		
		#dist is not a tensor, but a distribution object
		dist = torch.distributions.Categorical(probs)
		
		#action is a pytorch scalar with value 0, 1, 2, ...
		action = dist.sample()		

		#log_prob is log of probability associated with selected softmax action / item 
		#log_prob is a pytorch scalar	
		log_probs.append(dist.log_prob(action))
	
		#reward is scalar => torch.tensor(reward); rewards is a list of scalars => torch.tensor(rewards)
		obs_next, reward, terminated, truncated, info = env.step(action.item())
		obs_next_tensor = torch.tensor(obs_next, dtype=torch.float32)
		
		done = terminated or truncated 
		total_reward += reward

		#actor
		rewards.append(reward)

		#critic
		value = a2cCritic(obs_tensor)

		#don't track gradients for any operations inside this block
		with torch.no_grad():
			#value_next is pure inference / no need to build computational graph 
			value_next = a2cCritic(obs_next_tensor)
			#target and advantage are treated as constants. there is no need to back propagate through them.  
			#td(0) updates yield solutions with large biases, you are then bootstrapping on the biased result.
			#this is visible, when the reinforce and a2c_td0 landings are compared side by side.   
			target = torch.tensor(reward) + (1 - done) * gamma * value_next
			#can also do (target - value).detach()
			advantages.append(target - value)
	 
		#gradient is 2 * (target - value), for mse
		#target is fixed.  However, you do want a gradient to flow back through value, since this is the critic's net.  
		#i.e. value is a tensor output by the critic model.  backpropagate this: Value -> critic model parameters -> obs
		critic_loss = (target - value).pow(2)
		critic_list.append(critic_loss)

		obs_tensor = obs_next_tensor

	for i in range(len(log_probs)):
		log_prob = log_probs[i]
		G = advantages[i]
		policy_gradient = -log_prob * G
		actor_list.append(policy_gradient)

	#computational graph will back propagate the appropriate time step gradient updates for actor
	#this is similar to reinforce. except now, we are subtracting a critic baseline to reduce the variance.  
	a2cActor_estimate = torch.stack(actor_list, dim=0).mean()

	optimA2CActor.zero_grad()
	a2cActor_estimate.backward()
	optimA2CActor.step()

	#computational graph will back propagate the appropriate time step gradient updates for critic
	#gradient update is the gradient associated with mse_loss. 
	a2cCritic_loss = torch.stack(critic_list, dim=0).mean()

	optimA2CCritic.zero_grad()
	a2cCritic_loss.backward()
	optimA2CCritic.step()

	print(f"Episode {episode}, reward_received {reward:.2f}, total reward {total_reward:.2f},  a2cActor_estimate {a2cActor_estimate:.5f}, a2cCritic_loss {a2cCritic_loss:.5f}")

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
