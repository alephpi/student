import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import scipy.io as sio

def sigmoid(x: np.ndarray) -> np.ndarray:
	return 1.0 / (1.0 + np.exp(-x))

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

def plot_BAD(x: np.ndarray, save_to = None):
	try:
		x = x.reshape(20,16)
	except:
		raise ValueError(f'cannot convert to binary alpha digits format, need to be (20,16)!')
	plt.figure(figsize=(1,1))
	plt.imshow(x, cmap='gray', aspect='auto')
	if save_to != None:
		plt.savefig(save_to)

def generate_image(model, n_samples: int, n_gibbs: int):
	"""
	Args:
			n_gibbs (int): number of iterations in gibbs sampling
	"""
	for j in range(n_samples):
		x_ = model.generate(n_gibbs)
		plot_BAD(x_)