from typing import Tuple
import numpy as np
from tqdm import tqdm, trange
from utils import *
from .RBM import RBM
class DBN():

	def __init__(self, v_dim: int, h_dims: List[int], k: int =1, lr=0.1) -> None:
		'''
		Args:
			v_dim: dimension of visible units
			h_dim: dimension of hidden units
			k: times of Gibbs sampling
		'''
		super().__init__()
		assert h_dims != [], f'h_dims should at least contains one element'
		assert k > 0, f'k should be positive integer'
		self.k = k
		self.lr = lr
		self.rbms: list[RBM] = []
		self.rbms.append(RBM(v_dim=v_dim, h_dim=h_dims[0], k=k, lr=lr))
		for i in range(len(h_dims)-1):
			self.rbms.append(RBM(v_dim=h_dims[i], h_dim=h_dims[i+1], k=k, lr=lr))
		self.num_layers = len(self.rbms)

	def fit(self, data: np.ndarray, batch_size: int, num_epochs=100) -> None:
		"""train DBF layerwisely

		Args:
				data (np.ndarray): 2D array (n_samples, n_features)
		"""

		assert data.ndim == 2, f'data should be a 2D-array.'
		x = data.copy()
		for rbm in tqdm(self.rbms, desc='layer'):
			rbm.fit(x, batch_size=batch_size, num_epochs=num_epochs)
			_, x = rbm.v_to_h(x)
	
	def generate(self, n_gibbs: int) -> np.ndarray:
		"""data generation

		Args:
				n_gibbs (int): iteration times of gibbs sampling

		Returns:
				np.ndarray: generated data
		"""
		rbm = self.rbms[-1]
		x = rbm.generate(n_gibbs)
		for rbm in reversed(self.rbms[:-1]):
			_, x = rbm.h_to_v(x)
		return x