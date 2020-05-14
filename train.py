import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from model import VRNN


import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def loss_funct(out, x):
	prior_mu, prior_sig, decoder_mu, decoder_sig, x_decoded = out
	loss = 0.
	for i in range(x.shape[1]):

		#KL div
		a = Norm.Normal(prior_mu[i], prior_sig[i])
		b = Norm.Normal(decoder_mu[i], decoder_sig[i])
		kl_div = torch.mean(KL.kl_divergence(a, b))

		crossent = torch.mean(F.binary_cross_entropy(x_decoded[i], x[:,i,:], reduction = 'none'))
		loss += crossent + kl_div

	return loss



def train(loader, model, optimizer, epochs=100):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	for epoch in range(epochs):
		train_loss = 0
		for batch_idx, (data, target) in enumerate(loader):

			data = data.squeeze(1)
			data = (data/255).to(device)
			outs = model(data)
			loss = loss_funct(outs, data)
			model.zero_grad()
			loss.backward()
			_ = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
			optimizer.step()
			print(loss)


#hyperparameters
x_dim = 28
h_dim = 100
z_dim = 16
n_epochs = 100
clip = 10
learning_rate = 1e-3
batch_size = 512
seed = 128
print_every = 100
save_every = 10

#manual seed
torch.manual_seed(seed)
plt.ion()

#init model + optimizer + datasets
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data',
														  train=True,
														  download=True,
														  transform=transforms.ToTensor()),
										   batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('data',
														 train=False,
														 transform=transforms.ToTensor()),
										  batch_size=batch_size, shuffle=True)

model = VRNN(x_dim, h_dim, z_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#training + testing
train(train_loader, model, optimizer)
