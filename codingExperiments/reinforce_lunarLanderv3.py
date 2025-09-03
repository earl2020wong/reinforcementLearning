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
num_episodes = 2000 



class Reinforce_Policy(nn.Module):
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



def compute_returns(rewards, gamma):
	returns = []
	G = 0.0
	for r in reversed(rewards):
		G = r + gamma * G
		returns.insert(0,G)
	return returns



#training loop
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

reinforcePolicy = Reinforce_Policy(observation_dim, action_dim)
optimizer = optim.Adam(reinforcePolicy.parameters(), lr=learning_rate)



resume_training = False #True
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
	
	#obs is a numpy array => torch.tensor(obs)
	obs, _ = env.reset()
	obs_tensor = torch.tensor(obs, dtype=torch.float32)

	done = False
	total_reward = 0.0

	while not done: 
		#action is not a pytorch tensor, but just an index
		#action = env.action_space.sample()
		
		#probs is 1D torch.tensor with n elements 
		probs = reinforcePolicy(obs_tensor)
		
		#dist is not a tensor, but a distribution object
		dist = torch.distributions.Categorical(probs)
		
		#action is a pytorch scalar with value 0, 1, 2, ...
		action = dist.sample()		
		
		#log_prob is log of probability associated with selected softmax action / item 
		#log_prob is a pytorch scalar	
		log_probs.append(dist.log_prob(action))
	
		#reward is scalar => torch.tensor(reward); rewards is a list of scalars => torch.tensor(rewards)
		obs, reward, terminated, truncated, info = env.step(action.item())
		obs_tensor = torch.tensor(obs, dtype=torch.float32)
		
		done = terminated or truncated 
		total_reward += reward

		#time.sleep(0.05) 

		rewards.append(reward)

	returns = compute_returns(rewards, gamma)
	returns = torch.tensor(returns, dtype=torch.float32)
	#normalize each entry in the list to prevent large "outlier" scaling
	#i.e. (entry - meanList) / stdList 
	returns = (returns - returns.mean()) / (returns.std() + 1e-8)

	policy_gradient_list = []
	#rPL = [-log_prob * G for log_prob, G in zip(log_probs, returns)]

	for i in range(len(log_probs)):
		log_prob = log_probs[i]
		G = returns[i]

		#\nabla J(\theta) = E_{\pi_{\theta}}[\nabla_{\theta}log(\pi_{\theta}(a|s) * R]
		#increase the prob of actions that lead to higher return
		#decrease the prob of actions that lead to lower return 
		#log(large probability) = ~0; log(small probability) = large negative number
		
		#if prob of action is low and return is high (positive), gradient is strong and positive.
		#hence, really encourage this action. 
		#if prob of action is high and return is high (positive), gradient is less strong and positive.  
		# hence, gently encourage this action, as the probability is already high. 
		#in either case, push the policy to make the action more likely
		
		#if the prob of action is low and return is high (negative), gradient is strong and negative. 
		#hence, really discourage this action.  
		#if prob of action is high and return is high (negative), gradient is less strong and negative. 
		#hence, gently discourage this action.
		#in either case, push the policy to make the action less likely
		
		#computational graph says to push THE GRADIENTS associated with these updates for the various time steps.  
		policy_gradient = -log_prob * G
		#this is a list of torch tensors that are scalars.  
		#i.e. each element in the list has shape [], since the element is a torch scalar. 
		policy_gradient_list.append(policy_gradient)

	#torch.stack converts a list of pytorch scalar tensors to a 1D pytorch tensor with N elements.  i.e. shape becomes [N].  
	#sum and mean only work for pytorch tensors.
	#sum aggregates "loss" values, but not gradients 
	#mean aggregates "loss" values, but not gradients
	#if L is "sum" of loss values, then using mean produces a loss value of L / num_terms
	#if sum is used, each individual time step is back propagated "as is".  i.e. gradBP = G_t * \nabla_{\theta}\pi_{\theta}(a|s)
	#if mean is used, each individual time step is back propagated as gradBP / num_terms
	
	#in the end, .backward() will apply the gradient, gradBP or gradBP / num_terms, for every time step.    
   
	policy_gradient_estimate = torch.stack(policy_gradient_list).sum()

	optimizer.zero_grad()
	policy_gradient_estimate.backward()
	#\theta <- \theta + learning_rate * \nabla J(\theta) 
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
