import numpy as np
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
		self.h_dims = h_dims
		self.rbms: list[RBM] = []
		self.rbms.append(RBM(v_dim=v_dim, h_dim=h_dims[0]))
		for i in range(len(h_dims)-1):
			self.rbms.append(RBM(v_dim=h_dims[i], h_dim=h_dims[i+1]))

	def fit(self, data: np.ndarray, batch_size: int, num_epochs=100, k: int = 1, lr: int = 0.1) -> None:
		"""train DBF layerwisely

		Args:
				data (np.ndarray): 2D array (n_samples, n_features)
		"""

		assert data.ndim == 2, f'data should be a 2D-array.'
		for i, rbm in enumerate(self.rbms):
			rbm.fit(data, batch_size=batch_size, num_epochs=num_epochs, k=k, lr=lr, prog_bar_index = f'layer {i}')
			_, data = rbm.v_to_h(data)

	
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