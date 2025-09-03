import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm
import matplotlib.pyplot as plt  

import time
import math
import os
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#hyperparameters
learning_rate = 1e-4 #1e-3
batch_size = 128 #32
num_epochs = 201

num_time_steps = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_time_steps)

alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)  
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device) 

img_channels = 3
base_channels = 64
img_size = 32
freq_embedding_sample_points = 128



#Load images
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #scale to [-1, 1]
        torchvision.transforms.Lambda(lambda x: x * 2 - 1)]
)
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



class Positional_Embedding(nn.Module):
	def __init__(self, time_dim):
		super().__init__()
		self.time_dim = time_dim

	def forward(self, t):
		half_time_dim = self.time_dim // 2
		#create a time range of sample points that scales as log -> embedding_vector
		embedding_scale = math.log(10000) / (half_time_dim - 1)
		embedding_vector = torch.exp(torch.arange(half_time_dim) * -embedding_scale)

		embedding_vector = embedding_vector.to(t.device)

		#outer product multiplication 
		embedding_2D_sample = t[:, None].float() * embedding_vector[None, :]
		#dimensions: t x time_dim
		embedding_fixed_definition_vector = torch.cat([torch.sin(embedding_2D_sample), torch.cos(embedding_2D_sample)], dim=1) 
			
		return embedding_fixed_definition_vector

class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, time_embed_dim):
		super().__init__()
		self.norm1 = nn.GroupNorm(1, in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
	
		self.norm2 = nn.GroupNorm(1, out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

		self.time_embedding_proj = nn.Linear(time_embed_dim, out_channels)

		if in_channels != out_channels:
				self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
		else:
			self.skip = nn.Identity()

	def forward(self, x, actual_time_embedding):
		h = F.relu(self.norm1(x))
		h = self.conv1(h)

		t = self.time_embedding_proj(actual_time_embedding).unsqueeze(-1).unsqueeze(-1)
		h = h + t

		h = F.relu(self.norm2(h))
		h = self.conv2(h)

		#skip connection 
		return h + self.skip(x)

class SelfAttention(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.norm = nn.GroupNorm(8, channels)
		self.query = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
		self.key = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
		self.value = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
		self.proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding =0)

	#channelwise self attention over spatial position; every pixel attends to every other pixel
	def forward(self, x):
		B, C, H, W = x.shape
		x_norm = self.norm(x)

		#apply 1x1 convolution; learn a representation for each pixel location
		#q, k, v are [B, C, H*W]
		#treat spatial locations as tokens in a sequence
		q = self.query(x_norm).reshape(B, C, -1)
		k = self.key(x_norm).reshape(B, C, -1)
		v = self.value(x_norm).reshape(B, C, -1)

		#q.permute(0, 2, 1) becomes [B, H*W, C]
		#k is [B, C, H*W]
		#q @ k = [B, H*W, H*W]
		#every positions compares to every other position; how similar one pixel is to another 
		attn = torch.bmm(q.permute(0, 2, 1), k)

		#normalize scores and get weights for each pixel 
		attn = attn / (C ** 0.5)
		attn = torch.softmax(attn, dim=-1)

		#multiply by v to get result / mix values according to attention
		out = torch.bmm(v, attn.permute(0, 2, 1))
		out = out.reshape(B, C, H, W)
		out = self.proj(out)

		#skip connection
		return x + out

class Bottleneck(nn.Module):
	def __init__(self, channels, time_embed_dim):
		super().__init__()
		self.residual1 = ResidualBlock(channels, channels, time_embed_dim)
		self.attention = SelfAttention(channels)
		self.residual2 = ResidualBlock(channels, channels, time_embed_dim)

	def forward(self, x, actual_time_embedding):
		x = self.residual1(x, actual_time_embedding)
		x = self.attention(x)
		x = self.residual2(x, actual_time_embedding)
		
		return x

class UNet1_5(nn.Module):
	def __init__(self, in_channels, base_channels, out_channels, time_embed_dim):
		super().__init__()
		self.time_mlp = nn.Sequential(
			Positional_Embedding(time_embed_dim), 
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.ReLU()
		)

		self.enc1 = ResidualBlock(in_channels, base_channels, time_embed_dim) 
		self.enc2 = ResidualBlock(base_channels, 2 * base_channels, time_embed_dim)
		self.down = nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=4, stride=2, padding=1)

		self.bottleneck = Bottleneck(2 * base_channels, time_embed_dim)

		self.up = nn.ConvTranspose2d(2 * base_channels, 2 * base_channels, kernel_size=4, stride=2, padding=1)
		self.dec2 = ResidualBlock(4 * base_channels, base_channels, time_embed_dim)
		self.dec1 = ResidualBlock(2 * base_channels, out_channels, time_embed_dim)

	def forward(self, x, t):
		time_embedding = self.time_mlp(t)

		e1 = self.enc1(x, time_embedding)
		e2 = self.enc2(e1, time_embedding)
		ds = self.down(e2)

		b = self.bottleneck(ds, time_embedding)

		us = self.up(b)
		us = torch.cat([us, e2], dim=1)
		u = self.dec2(us, time_embedding)
		u = torch.cat([u, e1], dim=1)
		out = self.dec1(u, time_embedding)

		return out
			
#previous model 		 
class Diffusion_Noise_Estimator(nn.Module):
	def __init__(self, img_channels, base_channels, freq_embedding_points):
		super().__init__()
		
		#def freq_mlp(t):
    			#out = SinusoidalPosEmb(...)(t)       # [B, freq_embedding_points]
    			#out = nn.Linear(...)(out)            # [B, freq_embedding_points]
    			#out = nn.ReLU()(out)
    			#out = nn.Linear(...)(out)
    			#return out
		#above method is performing the same functionality as the nn.Sequential code below	

		self.freq_embedding_mlp = nn.Sequential(
			#t = torch.tensor([10, 20, 30])             # shape: [3]
			#emb = Positional_Embedding(128)(t)
			#emb.shape -> torch.Size([3, 128])	
			Positional_Embedding(freq_embedding_points),
			nn.Linear(freq_embedding_points, freq_embedding_points),
			nn.ReLU(),
			nn.Linear(freq_embedding_points, freq_embedding_points)
		)
 		
		#kernel, stride, pad
		#stride 2 => 32x32 to 16x16 to 8x8
		self.enc1 = nn.Conv2d(img_channels, base_channels, kernel_size=3, stride=1, padding=1)
		self.egn1 = nn.GroupNorm(num_groups=8, num_channels=base_channels)
		self.enc2 = nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, stride=2, padding=1)
		self.egn2 = nn.GroupNorm(num_groups=8, num_channels=2 * base_channels)
		self.enc3 = nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=3, stride=2, padding=1) 
		self.egn3 = nn.GroupNorm(num_groups=8, num_channels=4 * base_channels)

#		self.bottleneck = nn.Sequential(
#			nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=3, stride=1, padding=1),
#			nn.GroupNorm(8, 4 * base_channels),
#			nn.ReLU(),
#			nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=3, stride=1, padding=1),
#			nn.GroupNorm(8, 4 * base_channels),
#		#	nn.ReLU()
#		)

		self.bottleneck = nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=3, stride=1, padding=1)
		self.bngn = nn.GroupNorm(num_groups=8, num_channels=4 * base_channels)

		#takes output of self.freq_embedding_mlp and projects it to the input dimension for self.bottleneck
		self.freq_projection_embedding = nn.Linear(freq_embedding_points, 4 * base_channels)

		# stride 2 => 16x16, 32x32 	
		self.dec3 = nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=4, stride=2, padding=1)
		self.dgn3 = nn.GroupNorm(num_groups=8, num_channels=2 * base_channels)		
		self.dec2 = nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=4, stride=2, padding=1)
		self.dgn2 = nn.GroupNorm(num_groups=8, num_channels=base_channels)
#		self.dec1 = nn.Sequential(
#			nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
#			nn.GroupNorm(8, base_channels),
#			nn.ReLU(),
#			nn.Conv2d(base_channels, img_channels, kernel_size=3, stride=1, padding=1),
#		#	nn.Tanh()
#		)
		
		self.dec1 = nn.Conv2d(base_channels, img_channels, kernel_size=3, stride=1, padding=1)

	def forward(self, x, t):
		#x: [B, 3, 32, 32]
		#t: [B] 

		#1)
		t = t.float() #/ num_time_steps

		freq_embed = self.freq_embedding_mlp(t)
		freq_embedding_projection = self.freq_projection_embedding(freq_embed)

		e1 = F.relu(self.egn1(self.enc1(x)))  	# [B, base_channels, 32, 32]
		e2 = F.relu(self.egn2(self.enc2(e1)))	# [B, base_channels * 2, 16, 16]
		e3 = F.relu(self.egn3(self.enc3(e2)))	# [B, base_channels * 4, 8, 8]
		#m.shape is [B, 4 * base_channels, H, W]
		m = F.relu(self.bngn(self.bottleneck(e3)))	# [B, base_channels * 4, 8, 8]

		# [:, :, None, None] reshapes [B, 4 * base_layer_channels] to [B, 4 * base_layer_channels, 1, 1]
		# broadcasting makes [B, 4 * base_layer_channels, H, W] + [B, 4 * base_layer_channels, 1, 1] - > [...., H, W]
		m = m + freq_embedding_projection[:, :, None, None]	

		d3 = F.relu(self.dgn3(self.dec3(m)))	# [B, base_channels * 2, 16, 16]	
		#skip connection between d3 and e2
		d3 = d3 + e2
		d2 = F.relu(self.dgn2(self.dec2(d3)))	# [B, base_channels * 4, 32, 32]
		#skip connection between d2 and e1
		d2 = d2 + e1
		#d1 = self.dec1(d2)		# [B, img_channels, 32, 32] 
		#2) torch.tanh
		
		#d1 = torch.tanh(self.dec1(d2))
		d1 = self.dec1(d2)
		
		return d1 



def add_noise(x0, t_index, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
	#creates random tensor of same shape as x0 [B, C, H, W], drawn from N(0,1)
	noise = torch.randn_like(x0)
	#t_index is time step index for each image in batch B
	#view(-1, 1, 1, 1)
	#reshapes index to [B, 1, 1, 1] 
	#= figure out what the dimension in the first location (batch) should be, 
	#given that the total number of elements remains the same. 
	sqrt_alpha = sqrt_alpha_cumprod[t_index].view(-1, 1, 1, 1)
	sqrt_one_minus_alpha = sqrt_one_minus_alpha_cumprod[t_index].view(-1, 1, 1, 1)

	#sqrt_alpha.shape torch.Size([32, 1, 1, 1])
	#print(f"sqrt_alpha.shape {sqrt_alpha.shape}")
	return ((sqrt_alpha * x0) + (sqrt_one_minus_alpha * noise)), noise

@torch.no_grad()
def sample_ddpm(model, n=16):
	model.eval()
	#x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
	x = torch.randn(n, 3, 32, 32).to(device)
	
	for t in reversed(range(1000)): #T)):
		#t_batch = torch.full((n,), t, device=DEVICE)
		t_batch = torch.full((n,), t, device=device)
		noise_pred = model(x, t_batch)
		beta = betas[t]
		alpha = alphas[t]
		alpha_hat = alphas_cumprod[t]
		#alpha_hat = alpha_cumprods[t]

		if t > 0:
			noise = torch.randn_like(x)
		else:
			noise = torch.zeros_like(x)

		x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * noise_pred) + torch.sqrt(beta) * noise
		
	return x

@torch.no_grad()
#2021 openAI paper - improved version of 2020 paper by Ho, et. al. 
#def Sample_DDIM(model, num_steps, img_size, batch_size, alpha_cumprod):   
#	model.eval()
#	model.to(device)
#
#	x = torch.randn(batch_size, 3, img_size, img_size).to(device) 
#	x_orig = x
#
#	ddim_steps = 50
#	t_schedule = np.linspace(num_steps-1, 0, ddim_steps, dtype=int)
#	for i in range(len(t_schedule) - 1): 
#		t = t_schedule[i]
#		t_next = t_schedule[i + 1]
#
#		t_batch = torch.full((batch_size,), t, dtype=torch.long).to(device)
#
#		#1)
#		#t_batch_normalized = t_batch.float() / num_steps 
#		#pred_noise is the \eps in the telescoping relationship
#		#pred_noise = model(x, t_batch_normalized) 
#		pred_noise = model(x, t_batch)
#
#		#think of x as first being the true noisy image x_{T}, then x_{T - 1}, then ...}
#		#we are then "inversing" the formula, to try and recover x0
#		
#		x0_est = (x - torch.sqrt(1 - alpha_cumprod[t]) * pred_noise) / torch.sqrt(alpha_cumprod[t]) 
#		#2)
#		#x0_est = torch.clamp(x0_est, -1.0, 1.0)  
#		#x0_est = torch.tanh(x0_est)
#
#		x = torch.sqrt(alpha_cumprod[t_next]) * x0_est + torch.sqrt(1 - alpha_cumprod[t_next]) * pred_noise
# 
#	return x_orig, x

def sample_ddm(model, num_steps, img_size, batch_size, betas, alphas, alpha_cumprod):
	model.eval()
	model.to(device)
	
	x_ddim = torch.randn(batch_size, 3, img_size, img_size).to(device)
	x_ddpm = x_ddim.clone()

	print(f"x_dim.shape:", x_ddim.shape)
	print(f"x_dim img_size:", img_size)

	ddim_steps = 50
	t_schedule = np.linspace(num_steps - 1, 0, ddim_steps, dtype=int)

	for i in range(len(t_schedule) - 1):
                t = t_schedule[i]
                t_next = t_schedule[i + 1]
                t_batch = torch.full((batch_size,), t, dtype=torch.long).to(device)
                pred_noise = model(x_ddim, t_batch)
                x0_est = (x_ddim - torch.sqrt(1 - alpha_cumprod[t]) * pred_noise) / torch.sqrt(alpha_cumprod[t])
                x_ddim = torch.sqrt(alpha_cumprod[t_next]) * x0_est + torch.sqrt(1 - alpha_cumprod[t_next]) * pred_noise 

	for t in reversed(range(num_steps)): 
                t2_batch = torch.full((batch_size,), t, device=device)
                noise_pred = model(x_ddpm, t2_batch)
                beta = betas[t]
                alpha = alphas[t]
                alpha_hat = alpha_cumprod[t]
                if t > 0:
                        noise = torch.randn_like(x_ddpm)
                else:
                        noise = torch.zeros_like(x_ddpm)
                x_ddpm = (1 / torch.sqrt(alpha)) * (x_ddpm - (1 - alpha) / torch.sqrt(1 - alpha_hat) * noise_pred) + torch.sqrt(beta) * noise

	return x_ddim, x_ddpm

def save_checkpoint(model, optimizer, epoch):
	torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
        },f"ddpm_checkpoint_{epoch}.pth")

def load_checkpoint(model, optimizer, filename="ddpm_checkpoint_7.pth"): 
	if os.path.isfile(filename):
		print(f"Loading checkpoint from {filename}...")
		checkpoint = torch.load(filename)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']	



diffusionModel = UNet1_5(img_channels, base_channels, img_channels, freq_embedding_sample_points).to(device)
#diffusionModel = Diffusion_Noise_Estimator(img_channels, base_channels, freq_embedding_sample_points).to(device)
optimizer = optim.Adam(diffusionModel.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in diffusionModel.parameters())
print(f"Total parameters: {total_params:,}")



resume_training = False 
if resume_training:
	checkpoint = torch.load("ddpm_checkpoint.pth")
	diffusionModel.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	start_epoch = checkpoint["epoch"] + 1
	print(f"Resuming training from epoch {start_epoch}")
else:
	start_epoch = 0 

#training loop
diffusionModel.train()
diffusionModel.to(device)

for epoch in range(start_epoch, num_epochs):
	total_loss = 0.0
	count = 0
	process_bar = tqdm.tqdm(loader)

	for x, _ in process_bar:
		x = x.to(device)
		#x.shape torch.Size([32, 3, 32, 32]), x.size(0) 32
		#print(f"x.shape {x.shape}, x.size(0) {x.size(0)}")

		#t.shape torch.Size([32])
		#print(f"t.shape {t.shape}")	
		#therefore, t_index contains 32 random integers between 0 and 1000
		t_index = torch.randint(0, num_time_steps, (x.size(0),)).long()
		t_index = t_index.to(x.device)
		print(f"x.device: ", x.device)

		x_t, noise = add_noise(x, t_index, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
		
		#pred_noise.shape torch.Size([32, 3, 32, 32])
                #print(f"pred_noise.shape {pred_noise.shape}")
		pred_noise = diffusionModel(x_t, t_index)
	
		loss = F.mse_loss(pred_noise, noise)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		count = count + 1
		total_loss = total_loss + loss.item()
		process_bar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f} avg total loss: {total_loss / count:.4f}")
		#Loss {total_loss / len(training_loader.dataset)}")
	
	if epoch % 10 == 0: 
		save_checkpoint(diffusionModel, optimizer, epoch)
		samples_ddim, samples_ddpm = sample_ddm(diffusionModel, num_time_steps, img_size, batch_size, betas, alphas, alphas_cumprod)

		sample_ddim = (samples_ddim[0].clamp(-1, 1) + 1) / 2.0
		save_image(sample_ddim, f"sample_ddim_new_epoch_{epoch}.png")

		sample_ddpm = (samples_ddpm[0].clamp(-1, 1) + 1) / 2.0
		save_image(sample_ddpm, f"sample_ddpm_new_epoch_{epoch}.png")

		diffusionModel.train()

time.sleep(5.0)
#load_checkpoint(diffusionModel, optimizer)

for i in range(10):
	print(f"i = {i}")
	diffusionModel.eval() 
	
	samples_ddim, samples_ddpm = sample_ddm(diffusionModel, num_time_steps, img_size, batch_size, betas, alphas, alphas_cumprod)

	#[-1, 1] to [0, 1] output range
	samplesC_ddim = (samples_ddim.clamp(-1, 1) + 1) / 2.0
	samplesC_ddpm = (samples_ddpm.clamp(-1, 1) + 1) / 2.0

	#convert for display: [1, 3, 32, 32] → [32, 32, 3]
	img_ddim = samplesC_ddim[0].permute(1, 2, 0).cpu().numpy() 
	img_ddpm = samplesC_ddpm[0].permute(1, 2, 0).cpu().numpy()

	plt.subplot(1, 2, 1)
	plt.imshow(img_ddim)
	plt.title("ddim")
	plt.axis('off')
	plt.subplot(1, 2, 2)
	plt.imshow(img_ddpm)
	plt.title("ddpm")
	plt.axis("off")
	
	#plt.show()
	plt.pause(1.0) 

	ddim_img_tensor = torch.from_numpy(img_ddim)
	ddpm_img_tensor = torch.from_numpy(img_ddpm)

	#ensure it's in the correct range and type
	ddim_tensor = ddim_img_tensor.float().permute(2, 0, 1)
	ddpm_tensor = ddpm_img_tensor.float().permute(2, 0, 1)

	#if needed, permute from [H, W, C] to [C, H, W]
	#if input_img_tensor.ndim == 3 and input_img_tensor.shape[-1] == 3:
#		input_img_tensor = input_img_tensor.permute(2, 0, 1)
#		img_tensor = img_tensor.permute(2, 0, 1)

	print(f"shape:", ddim_tensor.shape)
	print(f"type:", ddim_tensor.dtype)
	print(f"min:, max: ", ddim_tensor.min(), ddim_tensor.max())

	save_image(ddim_tensor, f"ddim_epoch_200_{i}.png")
	save_image(ddpm_tensor, f"ddpm_epoch_200_{i}.png")
