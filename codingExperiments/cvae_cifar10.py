import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn



#hyperparameters
learning_rate = 1e-3
latent_dim = 128
epochs = 100
batch_size = 128


#load images
transform = transforms.ToTensor()
training_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)



class Convolution_Variational_AutoEncoder(nn.Module):
	def __init__(self, latent_dim):
		super().__init__()

		#3 => 32 => 64 => 128 channels 
		#32x32 => 16x16 => 8x8 => 4x4
		
		#kernel, stride, pad
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), 
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
		)
		#make 2D array one big, long vector 
		#128 feature maps, of size 4x4
		self.flatten = nn.Flatten()

		self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
		self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
		self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)

		#reverse the process
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), 
			nn.Sigmoid(),
		)

	def encode(self, x):
		x = self.encoder(x)
		x = self.flatten(x)
		mu = self.fc_mu(x)
		logvar = self.fc_logvar(x)
		return mu, logvar

	def decode(self, z):
		x = self.fc_decode(z)
		#unflatten, -1 means to infer batch size automatically
		#then, convert 784 length vector to 128, 4x4 feature maps
		x = x.view(-1, 128, 4, 4)
		x = self.decoder(x)
		return x

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(mu)
		z = mu + eps * std
		return z
		
	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		x_reconstruct = self.decode(z)
		return x_reconstruct, mu, logvar



#training loop
cvaeModel = Convolution_Variational_AutoEncoder(latent_dim)
optimizer = torch.optim.Adam(cvaeModel.parameters(), lr=learning_rate)

for epoch in range(epochs):
	total_loss = 0

	cvaeModel.train()
	for imgs, _ in training_loader:
		reconstruct_images, mu, logvar = cvaeModel(imgs)
		
		#inherent blurry reconstruction result, due to vanilla mse_loss
		reconstruct_loss = torch.nn.functional.mse_loss(imgs, reconstruct_images, reduction='sum')
		#applying the known mathematical derivation 
		kl_divergence_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	
		#without kl_divergence_term, more of a true 1-1 reconstruction, since the distribution closeness 
		#constraint is not enforced	
		loss = reconstruct_loss + kl_divergence_term

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss = total_loss + loss.item()

	print(f"Epoch {epoch}, Loss R {reconstruct_loss}, KL Divergence Term {kl_divergence_term}, Loss {total_loss / len(training_loader.dataset)}")

	cvaeModel.eval()
	with torch.no_grad():
		imgs, _ = next(iter(training_loader))
		reconstructed_images, _, _ = cvaeModel(imgs)

		for k in range(4):
			plt.subplot(2, 4, k + 1)
			plt.imshow(imgs[k].permute(1, 2, 0))
			plt.axis('off')

			plt.subplot(2, 4, k + 1 + 4)

	if epoch % 5  == 0:
		save_image(imgs[0], f"cvae_orig_with_kl_epoch_{epoch}.png")
		save_image(reconstructed_images[0], f"cvae_reconstruct_with_kl_epoch_{epoch}.png")


with torch.no_grad():
	imgs, _ = next(iter(training_loader))
	reconstructed_images, _, _ = cvaeModel(imgs)

	for k in range(4):
		plt.subplot(2, 4, k + 1)
		plt.imshow(imgs[k].permute(1, 2, 0))
		plt.axis('off')

		plt.subplot(2, 4, k + 1 + 4)
		plt.imshow(reconstructed_images[k].permute(1, 2, 0))
		plt.axis('off')
		plt.pause(2)
plt.show()
