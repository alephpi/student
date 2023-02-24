from typing import Tuple
import numpy as np
from tqdm import tqdm, trange
from utils import *
class RBM():

	def __init__(self, v_dim: int, h_dim: int, k: int =1) -> None:
		"""
		Args:
				v_dim (int): dimension of visible layer
				h_dims (List[int]): dimensions of hidden layers
				lr (float, optional): learning rate. Defaults to 0.1.
		"""
		super().__init__()
		self.c = np.zeros(v_dim)
		self.b = np.zeros(h_dim)
		self.W = np.random.normal(0, 0.1, size=(v_dim, h_dim))
		assert k > 0, f'k should be positive integer'
		self.k = k
		self.v_dim = v_dim
		self.h_dim = h_dim

	def v_to_h(self, v: np.ndarray) -> np.ndarray:
		# correspond to entree_sortie_RBM
		"""
		Sampling hidden units conditional to visible units

		Args:
				v (np.ndarray): visible units

		Returns:
				np.ndarray: hidden units
		"""
		p = sigmoid(v @ self.W + self.b)
		samples = np.random.binomial(1, p=p)
		return p, samples

	def h_to_v(self, h: np.ndarray) -> np.ndarray:
		# correspond to sortie_entree_RBM
		"""
		Sampling hidden units conditional to visible units

		Args:
				h (np.ndarray): hidden units

		Returns:
				np.ndarray: visible units
		"""

		p = sigmoid(h @ self.W.T + self.c)
		samples = np.random.binomial(1, p=p)
		return p, samples

	def gibbs_sampling(self, v:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		ph, h = self.v_to_h(v)
		pv, v = self.h_to_v(h)
		return ph, h, pv, v

	def forward(self, v:np.ndarray, k:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		'''
		input: data point v_0
		output: data point v_0, k-th sample in gibbs chain v_k  
		'''
		ph_0, h_0 = self.v_to_h(v)
		h_k = h_0
		for _ in range(k):
			pv_k, v_k = self.h_to_v(h_k)
			ph_k, h_k = self.v_to_h(v_k)
		return v, ph_0, v_k, ph_k
	
	def update(self, v_0: np.ndarray, ph_0: np.ndarray, v_k: np.ndarray, ph_k: np.ndarray, batch_size: int, lr: int) -> None:
		"""weight update

		Args:
				x (np.ndarray): input data
		"""
		dW = v_0.T @ ph_0 - v_k.T @ ph_k
		dc = (v_0 - v_k).sum(axis=0)
		db = (ph_0 - ph_k).sum(axis=0)
		self.W += lr * dW / batch_size
		self.c += lr * dc / batch_size
		self.b += lr * db / batch_size

	def fit(self, data: np.ndarray, batch_size: int = 8, num_epochs=100, k: int =1, lr: int = 0.1, prog_bar=False) -> None:
		"""train RBM

		Args:
				data (np.ndarray): 2D array (n_samples, n_features)
				k (int): times of Gibbs sampling in Contrastive Divengence learning
		"""

		assert data.ndim == 2, f'data should be a 2D-array.'
		if prog_bar: 
			pbar = trange(num_epochs)
		else:
			pbar = range(num_epochs)
		x = data.copy()
		for i in pbar:
			np.random.shuffle(x)
			loss = 0
			for batch in range(0, x.shape[0], batch_size):
				x_batch = x[batch: batch+batch_size, :]
				v_0, ph_0, v_k, ph_k = self.forward(x_batch, k=k)
				self.update(v_0, ph_0, v_k, ph_k, batch_size=x_batch.shape[0], lr=lr)
				loss += np.linalg.norm(v_0 - v_k, ord='fro') ** 2
			loss /= x.size
			if prog_bar:
				pbar.set_postfix(l2_loss = loss)
	
	def generate(self, n_gibbs: int) -> np.ndarray:
		"""data generation

		Args:
				n_gibbs (int): iteration times of gibbs sampling

		Returns:
				np.ndarray: generated data
		"""
		noise = np.random.randint(size=self.v_dim,low=0,high=2)
		x = noise
		for i in range(n_gibbs):
			_, _, _, x = self.gibbs_sampling(x)
		return x