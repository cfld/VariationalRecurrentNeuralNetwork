import torch
import torch.nn as nn
import numpy as np

import torch.distributions.normal as norm
import torch.distributions.kl as kl
import torch.nn.functional as F


class VRNN(nn.Module):

	def __init__(self, x_dim, h_dim, z_dim):
		super(VRNN,self).__init__()

		self.x_dim = x_dim
		self.h_dim = h_dim
		self.z_dim = z_dim

		# feature extractors of x 4 hidden layer w relu
		self.x_fx = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()
		)

		# feature extractor of z 4 hidden layer w relu
		self.z_fx =  nn.Sequential(
			nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

		# eq 5 (phi_prior(h_t-1) -> mu/sig)
		self.prior_fx = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU()
		)

		self.prior_mean = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, z_dim),
		)

		self.prior_var = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, z_dim),
			nn.Softplus()
		)

		# eq 6, phi_dec(phi(z), h_t-1) -> mu/sig
		self.decoder_fx = nn.Sequential(
			nn.Linear(h_dim * 2, h_dim),
			nn.ReLU()
		)

		self.decoder_mean = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, x_dim),
			nn.Sigmoid()
		)

		self.decoder_var = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, x_dim),
			nn.Softplus()
		)

		# encoder phi_enc(phi_x, h) ->
		self.encoder_fx = nn.Sequential(
			nn.Linear(h_dim * 2, h_dim),
			nn.ReLU()
		)

		# VRE regard mean values sampled from z as the output
		self.encoder_mean = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, z_dim),
		)
		self.encoder_var = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, z_dim),
			nn.Softplus()
		)

		# using the recurrence equation to update its hidden state
		self.rnn = nn.GRUCell(h_dim * 2, h_dim)


	def forward(self, x):
		'''
		:param x: shape of [frame, bs, fts dim]
		:return:
		'''
		h = torch.zeros([x.shape[0], self.h_dim], device=x.device)
		prior_mu_all, prior_sig_all     = [], []
		encoder_mu_all, encoder_sig_all = [], []
		decoder_mu_all, decoder_sig_all = [], []

		for t in range(x.shape[1]):

			# extract x
			phi_x = self.x_fx(x[:,t,:])

			# priors
			prior_ft  = self.prior_fx(h)
			prior_mu  = self.prior_mean(prior_ft) # not used
			prior_sig = self.prior_var(prior_ft) # not used

			# encoder
			encoder_ft  = self.encoder_fx(torch.cat([phi_x, h], dim=1))
			encoder_mu  = self.encoder_mean(encoder_ft)
			encoder_sig = self.encoder_var(encoder_ft)

			# decoder
			z_sampled = self.reparam(encoder_mu, encoder_sig)
			phi_z     = self.z_fx(z_sampled)
			decoder_ft  = self.decoder_fx(torch.cat([phi_z, h], dim=1))
			decoder_mu  = self.decoder_mean(decoder_ft) # not used
			decoder_sig = self.decoder_var(decoder_ft) # not used

			prior_mu_all.append(prior_mu)
			prior_sig_all.append(prior_sig)
			encoder_mu_all.append(encoder_mu)
			encoder_sig_all.append(encoder_sig)
			decoder_mu_all.append(decoder_mu)

			h = self.rnn(torch.cat([phi_x, phi_z], dim=1), h)

		return [prior_mu_all, prior_sig_all, encoder_mu_all, encoder_sig_all, decoder_mu_all]



	def reparam(self, mu, var):
		epsilon = torch.rand_like(mu, device=mu.device)
		return mu+var*epsilon











