from model.DBN import DBN
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):

	def __init__(self, dbn:DBN, num_classes:int =10) -> None:
		"""init DNN with a DBN (pretrained or not)
		"""
		
		super().__init__()
		self.mlp = nn.Sequential()
		# init mlp layers as dbn layers
		for rbm in dbn.rbms:
			linear = nn.Linear(rbm.v_dim, rbm.h_dim)
			linear.weight.data = torch.tensor(rbm.W.T, dtype=torch.float32)
			linear.bias.data = torch.tensor(rbm.b, dtype=torch.float32)
			self.mlp.append(linear)
		self.out = nn.Linear(dbn.rbms[-1].h_dim, num_classes)

	def forward(self, x: torch.tensor) -> torch.tensor:
		# partially correspond to entree_sortie_reseau
		for linear in self.mlp:
			x = linear.forward(x)
			x = torch.sigmoid(x)
		x = self.out(x)
		return F.log_softmax(x, dim=1)