import gym 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from collections import deque
import random
import time
import pickle
import os



print(gym.envs.registry.keys())
#env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")



#hyperparameters 
learning_rate = 1e-3
gamma = 0.99

eps_begin =  1.0
eps_end = 0.1
eps_decay = 1000000
target_replace_after_n_iterations = 10000

stack_size = 4
batch_size = 32
buffer_size = 100000

max_num_episodes = 2005
min_buffer_length_before_training_can_begin = 10000



class Replay_Buffer:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)
	
	def push(self, ob, action, reward, next_ob, done):
		self.buffer.append((ob, action, reward, next_ob, done))

	def sample_minibatch(self, batch_size):
		obs, actions, rewards, next_obs, dones = zip(*random.sample(self.buffer, batch_size))

		return (
			torch.tensor(np.array(obs), dtype=torch.float32),
			torch.tensor(np.array(actions), dtype=torch.int64),
			torch.tensor(np.array(rewards), dtype=torch.float32), 
			torch.tensor(np.array(next_obs), dtype=torch.float32), 
			torch.tensor(np.array(dones), dtype=torch.float32)
		)	

	def len_buffer(self):
		return len(self.buffer)

class DQN(nn.Module):
	def __init__(self, input_dims, num_actions):
		super().__init__()

		self.net = nn.Sequential(
			nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3136, 512),
			nn.ReLU(),
			nn.Linear(512, num_actions)
		)

	def forward(self, x):
		return self.net(x)

	

def preprocess_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        size8484 = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
        normIntensity = (size8484 / 255.0).astype(np.float32)

        return normIntensity

def create_or_append_stacked_frames(stacked_frames, new_frame, is_new_episode):
	#assert new_frame.shape == (84, 84), f"unexpected frame shape: {new_frame.shape}"
	#assert new_frame.dtype == np.float32, f"unexpected frame type: {new_frame.dtype}"
	if is_new_episode:
		frame_copies = []

		for _ in range(stack_size):
			copied_frame = new_frame.copy()
			frame_copies.append(copied_frame)

		stacked_frames = deque(frame_copies, maxlen=stack_size)

	else:   
		stacked_frames.append(new_frame)

	#for i, f in enumerate(stacked_frames):
	#	print(f"Frame {i}: shape {f.shape}, dtype={f.dtype}")

	#np.stack(stacked_frames, axis=0) returns a numpy construct, whose shape is (4, 84, 84) 
	#torch.tensor(np.stack(stacked_frames, axis=0)).unsqueeze(0) then makes it consumable by pytorch cnn input interface
	#stacked_frames is a deque, with 4 frames: deque([frame0, frame1, frame2, frame3]) 
	return np.stack(stacked_frames, axis=0), stacked_frames

def save_checkpoint(episode, model, optimizer, epsilon, replay_buffer, filename="dqn_checkpoint.pth", replay_filename="replay_buffer.pkl"):
	torch.save({
        	'episode': episode,
        	'model_state_dict': model.state_dict(),
        	'optimizer_state_dict': optimizer.state_dict(),
        	'epsilon': epsilon,
    	}, filename)

	#save replay buffer with pickle
	with open(replay_filename, "wb") as f:
        	pickle.dump(replay_buffer, f)
	
def load_checkpoint(model, optimizer, filename="dqn_checkpoint.pth", buffer_file="replay_buffer.pkl"): 
	if os.path.isfile(filename):
		print(f"Loading checkpoint from {filename}...")
		checkpoint = torch.load(filename)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		episode = checkpoint['episode']	
		epsilon = checkpoint['epsilon']
		print(f"Resuming from episode {episode} with epsilon {epsilon}")
		model.train()

		if os.path.isfile(buffer_file):
			with open(buffer_file, "rb") as f:
				replay_buffer = pickle.load(f)
			print(f"Loaded replay buffer with transitions.")
		else:
			replay_buffer = Replay_Buffer(buffer_size)
			print("No replay buffer found.  Created new one.")

		return episode, epsilon, replay_buffer 
	else: 
		print("No checkpoint found.  Starting fresh.")
		replay_buffer = Replay_Buffer(buffer_size)
		print("No replay buffer found.  Created new one.")

		return 0, 1.0, replay_buffer



#training loop
#observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

onPolicyNet = DQN((stack_size, 84, 84), action_dim)
targetNet = DQN((stack_size, 84, 84), action_dim)
targetNet.load_state_dict(onPolicyNet.state_dict())
#eval switches targetNet to eval mode.  i.e. ignores dropout, batchnorm, etc. if it is present in targetNet
targetNet.eval()

optimizer = optim.Adam(onPolicyNet.parameters(), lr=learning_rate)
replayBuffer = Replay_Buffer(buffer_size)



stacked_frames = deque(maxlen=stack_size)
steps_taken = 0
target_updates = 0

start_episode, epsilon, replayBuffer = load_checkpoint(onPolicyNet, optimizer)

for episode in range(start_episode, max_num_episodes): 
	ob = env.reset()[0]
	ob_frame = preprocess_frame(ob)
	#very first time, corresponding to the start of a new episode. create an stack of 4 identical frames
	np_stacked_frames, stacked_frames = create_or_append_stacked_frames(stacked_frames, ob_frame, True)
	episode_reward = 0
	done = False

	#for t in range(max_steps_in_episode):
	while not done:
		#graphical output 
		frame = env.render()		
		cv2.imshow("Breakout", frame)
		if cv2.waitKey(1) == ord('q'):
			break
		time.sleep(1/60) 

		epsilon = eps_end + (eps_begin - eps_end) * np.exp(-steps_taken / eps_decay)
		steps_taken = steps_taken + 1

		#exploration versus exploitation
		if random.random() < epsilon:
			action = env.action_space.sample()
		else: 
			with torch.no_grad():
				#creating a batch size of 1 for inference
				ob_tensor = torch.from_numpy(np_stacked_frames).unsqueeze(0).float()
				q_values = onPolicyNet(ob_tensor)
				action = q_values.argmax().item()

				#frame skipping.  Minh says that this helps training convergence.  
				total_reward = 0
				for _ in range(4):  #repeat action for 4 steps
    					next_ob, reward, terminated, truncated, _ = env.step(action)
					done = terminated or truncated
    					total_reward += reward
    					if done:
        					break

		#next_ob, reward, terminated, truncated, _ = env.step(action)
		#done = terminated or truncated
		next_frame = preprocess_frame(next_ob)
		next_np_stacked_frames, next_stacked_frames = create_or_append_stacked_frames(stacked_frames, next_frame, done)

		#replayBuffer.push(np_stacked_frames, action, reward, next_np_stacked_frames, done)
		replayBuffer.push(np_stacked_frames, action, total_reward, next_np_stacked_frames, done)

		np_stacked_frames = next_np_stacked_frames
		stacked_frames = next_stacked_frames

		episode_reward += reward

		if replayBuffer.len_buffer() > min_buffer_length_before_training_can_begin:
			#all parameters are returned as torch tensors, by the sample_minibatch method 
			obs, actions, rewards, next_obs, dones = replayBuffer.sample_minibatch(batch_size)

			#q_values_policy = onPolicyNet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
			q_values_policy_all = onPolicyNet(obs) #[32, action_dim]
			actions_unsqueeze = actions.unsqueeze(1) #[32] -> [32,1]
			q_values_policy_selected = q_values_policy_all.gather(1, actions_unsqueeze) #[32,1]
			q_values_policy = q_values_policy_selected.squeeze(1) # [32], (32,)
			
			with torch.no_grad():
				#max_q_values_target_next_selected = target(next_obs).max(1)[0] 	
				
				q_values_target_all_next = targetNet(next_obs)
				max_q_values_target_next_selected, max_q_indices_selected = q_values_target_all_next.max(dim=1)
				q_target = rewards + gamma * max_q_values_target_next_selected * (1 - dones)

			loss = torch.nn.functional.mse_loss(q_values_policy, q_target)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if steps_taken % target_replace_after_n_iterations == 0: 		
			targetNet.load_state_dict(onPolicyNet.state_dict())
			target_updates = target_updates + 1

		if done: 
			break

	print(f"Episode {episode}, episode reward {episode_reward:.5f}, target updates {target_updates}, replayBuffer len {replayBuffer.len_buffer()}")

	if episode % 50 == 0:
    		save_checkpoint(episode, onPolicyNet, optimizer, epsilon, replayBuffer)

env.close()

