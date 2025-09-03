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
learning_rate = 1e-4
gamma = 0.99
lam = 0.95
num_episodes = 2005
rollout_length = 4096 
eps_clip = 0.2

#we don't do polyak averaging or swapping of the model parameters.
#instead, we start by performing a long rollout of 4096 steps (state, action, reward). 
#actions are determined using the "old" policy / actor.
    
#next, we repeatedly apply the rollout results for num_epochs = 5, updating the 
#policy / actor and critic.  
#the data "input" is not temporally correlated though, since we (first) randomize the data from all 4096 steps.  
#next, we sample consecutive batches - [0:31], [32:63], ... , [4095-32: 4095] for processing during the epoch.  
#the policy / actor and critic are updated for every batch.  
#the same rollout data is applied a total of num_epochs = 5 times.  
#but, randomization of the rollout data occurs at the start of each new epoch.  
    
#upon completion of num_epochs = 5, we then perform a new rollout of 4096 steps, and start all over. 
#the "old" policy / actor and critic are now automatically updated with the policy / actor and critic trained 
#during num_epochs = 5
       
num_epochs = 5 
batch_size = 32
entropy_coeff = 0.01



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
			#nn.Softmax(dim=-1)
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
	GAE = 0.0
	G = 0.0

	values = values + [next_value]

	for t in reversed(range(len(rewards))):
		deltaGAE = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
		GAE = deltaGAE + gamma * lam * GAE * (1 - dones[t])
		advantages.insert(0, GAE)

	#	G = rewards[t] + gamma * G * (1 - dones[t]) 
	#		returns.insert(0, G)
	#5)
	returns = advantages + values[:-1]

	return returns, advantages



#training loop
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

ppoActor = PPO_Actor(observation_dim, action_dim)
optimPPOActor = optim.Adam(ppoActor.parameters(), lr=learning_rate)

ppoCritic = PPO_Critic(observation_dim)
optimPPOCritic = optim.Adam(ppoCritic.parameters(), lr=learning_rate) 

resume_training = True #False #True
if resume_training:
	checkpoint = torch.load("ppo_4096rollout_checkpoint.pth")
	ppoActor.load_state_dict(checkpoint["actor_state_dict"])
	optimPPOActor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
	ppoCritic.load_state_dict(checkpoint["critic_state_dict"])
	optimPPOCritic.load_state_dict(checkpoint["optimizer_critic_state_dict"])
	start_episode = checkpoint["episode"] + 1
	print(f"Resuming training from episode {start_episode}")
else: 
	start_episode = 0

for episode in range(start_episode, num_episodes):
	#oldPPOActor = PPO_Actor(observation_dim, action_dim)
	#oldPPOActor.load_state_dict(ppoActor.state_dict()) 
	
	obsR = []
	actions = []	
	rewards = []
	dones = []
	log_probs_old = []
	values = []

	total_reward = 0.0; 

	#obs is a numpy array => torch.tensor(obs)
	obs, _ = env.reset()
	obs = torch.tensor(obs, dtype=torch.float32)

	#perform a very long rollout.  if done results, during the rollout, then just "reset" and continue 
	#until 4096 steps have occurred. 
	#we are doing this, in lieu of creating a replay buffer. 
	
	#policy actions correspond to the "old" policy.  
	#i.e. there is a naturally created "policy gap" between the 4096 step rollout and the updated policy 
	#resulting from the application of num_epochs = 5 back propagated updates.    
	for _ in range(rollout_length): 
		#action is not a pytorch tensor, but just an index
		#action = env.action_space.sample()
	
		with torch.no_grad():	
			#probs is 1D torch.tensor with n elements 
			#probs = ppoActor(obs) #oldPPOActor(obs)
			#3) for an unexplained reason, it is recommended that logits are used with PPO.
			#supposedly, greater stability results. 
			#with logits, the softmax is now computed in Categorical
			logits = ppoActor(obs)

			#dist is not a tensor, but a distribution object
			#dist = torch.distributions.Categorical(probs)
			dist = torch.distributions.Categorical(logits=logits)

			#action is a pytorch scalar with value 0, 1, 2, ...
			action = dist.sample()		

			#log_prob is log of probability associated with selected softmax action / item 
			#log_prob is a pytorch scalar	
			log_prob_old = dist.log_prob(action) 

			value = ppoCritic(obs)

		#obs_next is numpy array => torch.tensor(obs_next)
		#reward is scalar => torch.tensor(reward), list of scalars => torch.tensor(rewards)
		obs_next, reward, terminated, truncated, info = env.step(action.item())
		obs_next = torch.tensor(obs_next, dtype=torch.float32)
		done = terminated or truncated		

		total_reward += reward

		obsR.append(obs) 
		actions.append(action)
		rewards.append(reward)
		dones.append(float(done))
		log_probs_old.append(log_prob_old)
		values.append(value)
	
		obs = obs_next

		if done: 
			obs, _ = env.reset()
			obs = torch.tensor(obs, dtype=torch.float32)

	with torch.no_grad():
		next_value = ppoCritic(obs)

	returns, advantages = compute_GAE(next_value, rewards, values, dones, gamma, lam)

	obsR = torch.stack(obsR)
	actions = torch.stack(actions)
	log_probs_old = torch.stack(log_probs_old)
	values = torch.stack(values)
	returns = torch.tensor(returns, dtype=torch.float32) 
	advantages = torch.tensor(advantages, dtype=torch.float32)

	#normalize
	returns = (returns - returns.mean()) / (returns.std() + 1e-4)
	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

	checkpoint = {
                "episode": episode,
                "actor_state_dict": ppoActor.state_dict(),
                "critic_state_dict": ppoCritic.state_dict(),
                "optimizer_actor_state_dict": optimPPOActor.state_dict(),
                "optimizer_critic_state_dict": optimPPOCritic.state_dict()
        }

	torch.autograd.set_detect_anomaly(True)
	
	#now, apply the rollout.  the rollout should have a lot of new starts from lunar lander, as 'done' was probably
	#hit many times, during the 4096 steps.
	#i.e. our rollout can be viewed as sampling from a replay buffer with variable length entries. 
	#each entry in the replay buffer corresponds to one complete session of lunar lander.
	#the 4096 step rollout is then created by concatenating the variable length replay buffer entries.    
	for _ in range(num_epochs):
		idx = np.arange(len(obsR))
		np.random.shuffle(idx)

		for start in range(0, len(obsR), batch_size):
			end = start + batch_size
			batch_idx = idx[start:end]
			
			#1) indentation error 1
			batch_obsR = obsR[batch_idx]
			batch_actions = actions[batch_idx]
			batch_log_probs_old = log_probs_old[batch_idx]
			batch_returns = returns[batch_idx]
			batch_advantages = advantages[batch_idx]

			#2) indentation error 2
			valuesR = ppoCritic(batch_obsR)
			#probs = ppoActor(batch_obsR)
			#dist = torch.distributions.Categorical(probs)

			#3)
			logits = ppoActor(batch_obsR)
			dist = torch.distributions.Categorical(logits=logits)

			#4)
			entropy = dist.entropy().mean()

			#actions = dist.sample()		
			log_probs_new = dist.log_prob(batch_actions)

			ratios = (log_probs_new - batch_log_probs_old).exp()

			surrogateNoClip = ratios * batch_advantages
			surrogateClip = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages

			policy_gradient = -torch.min(surrogateNoClip, surrogateClip)
			critic_loss = torch.nn.functional.mse_loss(valuesR, batch_returns)
		
			ppoActor_estimate = policy_gradient.mean() - entropy_coeff * entropy

			optimPPOActor.zero_grad()
			ppoActor_estimate.backward()
			optimPPOActor.step()

			ppoCritic_loss = critic_loss.mean()

			optimPPOCritic.zero_grad()
			ppoCritic_loss.backward()
			optimPPOCritic.step()

	print(f"Episode {episode}, total_reward during rollout / rollout_length {(total_reward / rollout_length):.2f}, policy_gradient.mean() {policy_gradient.mean()}, entropy.mean() {entropy}, ppoActor_estimate {ppoActor_estimate:.5f}, ppoCritic_loss {ppoCritic_loss:.2f}")

torch.save(checkpoint, "ppo_4096rollout_checkpoint.pth")
print("Checkpoint saved.")

env.close()
