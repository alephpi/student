import numpy as np
import matplotlib.pyplot as plt
from typing import List
import scipy.io as sio
def read_alpha_digit(data_file: str, labels: List[int] = None) -> np.ndarray:
	'''
	data_file: file path where binary alpha digits stores
	characters: list of characters that we want to extract from the dataset

	BAD dataset contains 39 different sample binary images of size (20,16) for 26 letters & 10 digits

	Return: a 2d array (36, 39 * 20 * 16)
	'''
	mat = sio.loadmat(data_file)
	res = [mat['dat'][char][i].flatten() for char in labels for i in range(39)]
	return np.array(res)

def plot_BAD(x: np.ndarray):
	assert x.shape == (20, 16), 'Wrong format for binary alpha digits!'
	plt.imshow(x, cmap='gray')
