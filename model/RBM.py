# adapt from https://github.com/bacnguyencong/rbm-pytorch/blob/master/rbm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
class RBM(nn.Module):

	def __init__(self, v_dim: int, h_dim: int, k: int) -> None:
		'''
		Args:
			v_dim: dimension of visible units
			h_dim: dimension of hidden units
			k: times of Gibbs sampling
		'''
		super().__init__()
		self.v_b = nn.Parameter(torch.zeros(v_dim))
		self.h_b = nn.Parameter(torch.zeros(h_dim))
		self.W = nn.Parameter(torch.empty((h_dim, v_dim)))
		self.W.data.normal_(0, 0.1)
		self.k = k

	def v_to_h(self, v: Tensor) -> Tensor:
		# correspond to entree_sortie_RBM
		'''
		Sampling hidden units conditional to visible units
		
		Args:
			v: visible units
		
		Returns:
			h: hidden units
		'''

		p = torch.sigmoid(F.linear(v, self.W, self.h_b))
		return p.bernoulli()

	def h_to_v(self, h: Tensor) -> Tensor:
		# correspond to sortie_entree_RBM
		'''
		Sampling hidden units conditional to visible units
		
		Args:
			h: hidden units
		
		Returns:
			v: visible units
		'''

		p = torch.sigmoid(F.linear(h, self.W.T, self.v_b))
		return p.bernoulli()

	def free_energy(self, v:Tensor) -> Tensor:
		# dim 0 is batch dimension
		v_term = torch.matmul(v, self.v_b.t())
		print(f'v_term.shape={v_term.shape}')
		w_x_h = F.linear(v, self.W, self.h_b)
		print(f'w_x_h.shape={w_x_h.shape}')
		# sum up along the dim 1
		h_term = torch.sum(F.softplus(w_x_h), dim=1)
		print(f'h_term.shape={h_term.shape}')
		return torch.mean(- h_term - v_term)

	def forward(self, v:Tensor):
		'''
		input: data point v_0
		output: data point v_0, k-th sample in gibbs chain v_k  
		'''
		h = self.v_to_h(v)
		for _ in range(self.k):
			v_gibb = self.h_to_v(h)
			h = self.v_to_h(v_gibb)
		return v, v_gibb