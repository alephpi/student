import numpy as np
import matplotlib.pyplot as plt
from typing import List
import scipy.io as sio
from torch import Tensor
from model.RBM import RBM
from model.DBN import DBN

class ThresholdTransform:
	def __init__(self, thr=0.5) -> None:
		self.thr = thr
	def __call__(self, x: Tensor):
		return (x > self.thr).to(x.dtype)

def sigmoid(x: np.ndarray) -> np.ndarray:
	return 1.0 / (1.0 + np.exp(-x))

def softmax(x: np.ndarray) -> np.ndarray:
	"""Compute softmax values for each sets of scores in x."""
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def read_alpha_digit(data_file: str, labels: List[int]) -> np.ndarray:
	"""read from Binary Alpha Digit dataset

	BAD dataset contains 39 different sample binary images of size (20,16) for 26 letters & 10 digits

	Args:
			data_file (str): file path where binary alpha digits stores
			labels (List[int]): list of characters that we want to extract from the dataset. 

	Returns:
			np.ndarray: a 2d array (num_labels, 39 * 20 * 16) 
	"""

	# Return: 
	# '''
	mat = sio.loadmat(data_file)
	x = []
	y = []
	for char in labels:
		for i in range(39):
			x.append(mat['dat'][char][i].flatten())
			y.append(char)
	return np.array(x), np.array(y) 

def plot_BAD(x: np.ndarray):
	plt.imshow(x, cmap='gray', aspect='auto')

def generate_image(model, n_samples: int, n_gibbs: int):
	"""
	Args:
			n_gibbs (int): number of iterations in gibbs sampling
	"""
	fig, ax = plt.subplots(1, n_samples,figsize=(n_samples, 1.25))
	for j in range(n_samples):
		x_ = model.generate(n_gibbs)
		ax[j].imshow(x_.reshape(20,16), cmap='gray', aspect='auto')
		ax[j].set_xticks([])
		ax[j].set_yticks([])
		ax[j].set_xticklabels([])
		ax[j].set_yticklabels([])
	return fig
	