from typing import Literal, Optional, List

import torch
from torch import Tensor, autograd
from torch.nn import Linear, CrossEntropyLoss
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from torch_geometric.nn.models import GraphSAGE
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import Adj

# according to pyG BasicGNN source code https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html
# and example https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
class Model(torch.nn.Module):
	def __init__(
		self,
		in_channels: int,
		hidden_channels: int,
		out_channels: int,
		num_layers: int = 3, #including the last linear one
		ewc_type : Optional[Literal['ewc','l2']] = None,
		ewc_lambda: float = 0,
	):
		super().__init__()

		self.sage = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers-1, bias=False)
		self.lin = Linear(in_features=hidden_channels, out_features=out_channels, bias=False)
		self.ewc_type = ewc_type
		self.ewc_lambda = ewc_lambda
		self.num_layers = num_layers

	def reset_parameters(self):
		self.sage.reset_parameters()
		self.lin.reset_parameters()

	def forward(self, data: Data) -> Tensor:
		x = self.sage(x=data.x, edge_index=data.edge_index)
		x = self.lin(x)
		return F.log_softmax(x, 1)

	def save(self, path):
		torch.save(self, path)
	
	# cross entropy loss
	def _classification_loss(self, data:Data) -> Tensor:
		y_pred = (self.forward(data)[:data.batch_size])[data.train_mask[:data.batch_size]]
		y_true = (data.y[:data.batch_size])[data.train_mask[:data.batch_size]]
		return F.cross_entropy(y_pred, y_true) 

	def _consolidation_loss(self) -> Tensor:
			losses = []
			for param_name, param in self.named_parameters():
					_buff_param_name = param_name.replace('.', '__')
					estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
					estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
					if self.ewc_type == 'l2':
							losses.append((10e-6 * (param - estimated_mean) ** 2).sum())
					else:
							losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
			return 1 * (self.ewc_lambda / 2) * sum(losses)

	def loss(self, data: Data) -> Tensor:
		if self.ewc_type is None:
			return self._classification_loss(data)
		# cross entropy loss + consolidation loss
		return self._classification_loss(data) + self._consolidation_loss()

	def _update_mean_params(self):
			for param_name, param in self.named_parameters():
					_buff_param_name = param_name.replace('.', '__')
					self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

	def _update_fisher_params(self, data: Data):
			log_likelihood = self._classification_loss(data)
			grad_log_liklihood = autograd.grad(log_likelihood, self.parameters())
			_buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters()]
			for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
					self.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)

	# not used, seems for testing
	def _save_fisher_params(self):
			for param_name, param in self.named_parameters():
					_buff_param_name = param_name.replace('.', '__')
					estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
					estimated_fisher = np.array(getattr(self, '{}_estimated_fisher'.format(_buff_param_name)))
					np.savetxt('estimated_mean', estimated_mean)
					np.savetxt('estimated_fisher', estimated_fisher)
					print(np.mean(estimated_fisher), np.max(estimated_fisher), np.min(estimated_fisher))
					break

	def register_ewc_params(self, data: Data):
			self._update_fisher_params(data)
			self._update_mean_params()

	# # for data visualization and evaluation
	# @torch.no_grad()
	# def inference(self, loader: NeighborLoader,
	# 							device: Optional[torch.device] = None,
	# 							progress_bar: bool = False) -> Tensor:
	# 		r"""Performs layer-wise inference on large-graphs using
	# 		:class:`~torch_geometric.loader.NeighborLoader`.
	# 		:class:`~torch_geometric.loader.NeighborLoader` should sample the the
	# 		full neighborhood for only one layer.
	# 		This is an efficient way to compute the output embeddings for all
	# 		nodes in the graph.
	# 		Only applicable in case :obj:`jk=None` or `jk='last'`.
	# 		"""
	# 		assert isinstance(loader, NeighborLoader)
	# 		assert len(loader.dataset) == loader.data.num_nodes
	# 		assert len(loader.node_sampler.num_neighbors) == 1
	# 		assert not self.training

	# 		if progress_bar:
	# 				pbar = tqdm(total=len(self.num_layers) * len(loader))
	# 				pbar.set_description('Inference')

	# 		x_all = loader.data.x.cpu() # loader.data.x here is a ndarray instead of torch Tensor
	# 		loader.data.n_id = torch.arange(x_all.size(0))

	# 		xs: List[Tensor] = []
	# 		for batch in loader:
	# 				x = x_all[batch.n_id].to(device)
	# 				edge_index = batch.edge_index.to(device)
	# 				x = self.sage(x, edge_index)[:batch.batch_size]
	# 				x = self.lin(x)[:batch.batch_size]
	# 				xs.append(x.cpu())
	# 				if progress_bar:
	# 						pbar.update(batch.batch_size)
	# 		x_all = torch.cat(xs, dim=0)
	# 		if progress_bar:
	# 				pbar.close()
	# 		return x_all