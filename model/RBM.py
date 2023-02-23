from typing import Tuple
import numpy as np
from tqdm import tqdm, trange
from utils import *
class RBM():

	def __init__(self, v_dim: int, h_dim: int, k: int =1, lr=0.1) -> None:
		'''
		Args:
			v_dim: dimension of visible units
			h_dim: dimension of hidden units
			k: times of Gibbs sampling
		'''
		super().__init__()
		self.v_b = np.zeros(v_dim)
		self.h_b = np.zeros(h_dim)
		self.W = np.random.normal(0, 0.1, size=(v_dim, h_dim))
		assert k > 0, f'k should be positive integer'
		self.k = k
		self.lr = lr

	def v_to_h(self, v: np.ndarray) -> np.ndarray:
		# correspond to entree_sortie_RBM
		"""
		Sampling hidden units conditional to visible units

		Args:
				v (np.ndarray): visible units

		Returns:
				np.ndarray: hidden units
		"""
		p = sigmoid(v @ self.W + self.h_b)
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

		p = sigmoid(h @ self.W.T + self.v_b)
		samples = np.random.binomial(1, p=p)
		return p, samples

	def gibbs_sampling(self, v:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		ph, h = self.v_to_h(v)
		pv, v = self.h_to_v(h)
		return ph, h, pv, v

	def forward(self, v:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		'''
		input: data point v_0
		output: data point v_0, k-th sample in gibbs chain v_k  
		'''
		ph_0, h_0 = self.v_to_h(v)
		h_k = h_0
		for _ in range(self.k):
			pv_k, v_k = self.h_to_v(h_k)
			ph_k, h_k = self.v_to_h(v_k)
		return v, ph_0, v_k, ph_k
	
	def update(self, v_0: np.ndarray, ph_0: np.ndarray, v_k: np.ndarray, ph_k: np.ndarray, batch_size: int) -> None:
		"""weight update

		Args:
				x (np.ndarray): input data
		"""
		dW = v_0.T @ ph_0 - v_k.T @ ph_k
		dv_b = (v_0 - v_k).sum(axis=0)
		dh_b = (ph_0 - ph_k).sum(axis=0)
		self.W += self.lr * dW / batch_size
		self.v_b += self.lr * dv_b / batch_size
		self.h_b += self.lr * dh_b / batch_size

	def fit(self, data: np.ndarray, batch_size: int, num_epochs=100) -> None:
		"""train RBM

		Args:
				data (np.ndarray): 2D array (n_samples, n_features)
		"""

		assert data.ndim == 2, f'data should be a 2D-array.'
		pbar = trange(num_epochs)
		x = data.copy()
		for i in pbar:
			np.random.shuffle(x)
			loss = 0
			for batch in range(0, x.shape[0], batch_size):
				x_batch = x[batch: batch+batch_size, :]
				v_0, ph_0, v_k, ph_k = self.forward(x_batch)
				self.update(v_0, ph_0, v_k, ph_k, batch_size=x_batch.shape[0])
				loss += np.linalg.norm(v_0 - v_k, ord='fro') ** 2
			loss /= x.size
			pbar.set_postfix(l2_loss = loss)
	
	def inference(self, x: np.ndarray, n_gibbs: int) -> np.ndarray:
		"""data generation

		Args:
				n_gibbs (int): iteration times of gibbs sampling

		Returns:
				np.ndarray: generated data
		"""
		x_ = x
		for i in range(n_gibbs):
			_, _, _, x_ = self.gibbs_sampling(x_)
		return x_