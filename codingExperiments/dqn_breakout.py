import gym 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from collections import deque
import random
import time



print(gym.envs.registry.keys())
#env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")



#hyperparameters 
learning_rate = 1e-3
gamma = 0.99

eps_begin =  1.0
eps_end = 0.1
eps_decay = 1000000
target_replace_after_n_iterations = 1000

stack_size = 4
batch_size = 32
buffer_size = 100000

max_num_episodes = 1000
max_steps_in_episode = 10000



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
	#	print(f"Frame {i}: shape-{f.shape}, dtype={f.dtype}")

	return np.stack(stacked_frames, axis=0), stacked_frames



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
ob = preprocess_frame(env.reset()[0])
#True creates a new stack with 4 identical ob entries
np_stacked_frames, stacked_frames = create_or_append_stacked_frames(stacked_frames, ob, True)



#create replay buffer entries
for _ in range(buffer_size):
	action = env.action_space.sample()
	next_ob, reward, done, _, _ = env.step(action)
	next_frame = preprocess_frame(next_ob)
	next_np_stacked_frames, next_stacked_frames = create_or_append_stacked_frames(stacked_frames, next_frame, done)
	replayBuffer.push(np_stacked_frames, action, reward, next_np_stacked_frames, done)

	if not done:
		np_stacked_frames = next_np_stacked_frames
	else: 
		ob = env.reset()[0]
		new_frame = preprocess_frame(ob)
		np_stacked_frames = create_or_append_stacked_frames(stacked_frames, new_frame, True)[0]



stacked_frames2 = deque(maxlen=stack_size)
steps_taken = 0

for episode in range(max_num_episodes):
	ob = env.reset()[0]
	ob_frame = preprocess_frame(ob)
	np_stacked_frames2, stacked_frames2 = create_or_append_stacked_frames(stacked_frames2, ob_frame, True)
	episode_reward = 0

	for t in range(max_steps_in_episode):
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
				ob_tensor = torch.tensor(np.array([np_stacked_frames2]), dtype=torch.float32)
				q_values = onPolicyNet(ob_tensor)
				action = q_values.argmax().item()

		next_ob, reward, done, _, _ = env.step(action)
		next_frame = preprocess_frame(next_ob)
		next_np_stacked_frames2, stacked_frames2 = create_or_append_stacked_frames(stacked_frames2, next_frame, False)

		replayBuffer.push(np_stacked_frames2, action, reward, next_np_stacked_frames2, done)

		np_stacked_frames2 = next_np_stacked_frames2
		episode_reward += reward

		#should always be true
		if replayBuffer.len_buffer() > batch_size:
			obs, actions, rewards, next_obs, dones = replayBuffer.sample_minibatch(batch_size)

			#q_values_policy = onPolicyNet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

			q_values_policy_all = onPolicyNet(obs) #(32, action_dim)
			actions_unsqueeze = actions.unsqueeze(1) #(32,1)
			q_values_policy_selected = q_values_policy_all.gather(1, actions_unsqueeze) #(32,1)
			q_values_policy = q_values_policy_selected.squeeze(1) # (32,)
			
			with torch.no_grad():
				#max_q_values_target_next_selected = target(next_obs).max(1)[0] 	
				
				q_values_target_all_next = targetNet(next_obs)
				max_q_values_target_next_selected, max_q_indices_selected = q_values_target_all_next.max(dim=1)
				q_target = rewards + gamma * max_q_values_target_next_selected * (1 - dones)

			loss = nn.MSELoss()(q_values_policy, q_target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if steps_taken % target_replace_after_n_iterations == 0: 		
			targetNet.load_state_dict(onPolicyNet.state_dict())

		if done: 
			break

	print(f"Episode {episode}, episode reward {episode_reward:.5f}")

env.close()

