from typing import Tuple
import numpy as np
from tqdm import tqdm, trange
from utils import *
from .RBM import RBM
class DBN():

	def __init__(self, v_dim: int, h_dims: List[int]) -> None:

		"""
		Args:
				v_dim (int): dimension of visible layer
				h_dims (List[int]): dimensions of hidden layers
				k (int, optional): times of Gibbs sampling in Contrastive Divengence learning
				lr (float, optional): learning rate. Defaults to 0.1.
		"""

		super().__init__()
		assert h_dims != [], f'h_dims should at least contains one element'
		self.rbms: list[RBM] = []
		self.rbms.append(RBM(v_dim=v_dim, h_dim=h_dims[0]))
		for i in range(len(h_dims)-1):
			self.rbms.append(RBM(v_dim=h_dims[i], h_dim=h_dims[i+1]))
		self.num_layers = len(self.rbms)

	def fit(self, data: np.ndarray, batch_size: int, num_epochs=100, k: int = 1, lr: int = 0.1) -> None:
		"""train DBF layerwisely

		Args:
				data (np.ndarray): 2D array (n_samples, n_features)
		"""

		assert data.ndim == 2, f'data should be a 2D-array.'
		x = data.copy()
		for rbm in tqdm(self.rbms, desc='layer'):
			rbm.fit(x, batch_size=batch_size, num_epochs=num_epochs, k=k, lr=lr)
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