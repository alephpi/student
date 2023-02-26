import numpy as np
import pickle
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from .DBN import DBN
from .DNN import DNN
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from typing import List
from utils import ThresholdTransform

class MNISTDNN(pl.LightningModule):
	def __init__(self,
	      h_dims:List[int],
				pretrain:bool,
				num_features=784,
				num_classes=10,
				lr: float =1e-3,
				batch_size=128,
				data_dir: str = None,
				data_size: int = 60000,
				pretrain_lr = 0.1,
				pretrain_epochs=100,
				load_pretrain_path: str = None,
				save_pretrain_path:str = None) -> None:


		"""MNIST DNN wrapper

		Args:
				h_dims (List[int]): dimensions of hidden layers of DBN
				pretrain (bool): pretrain the DBN or not
				num_features (int, optional): feature dimension of MNIST tensor. Defaults to 784.
				num_classes (int, optional): 10 digits classes. Defaults to 10.
				lr (float, optional): learning rate of training DNN. Defaults to 1e-3.
				batch_size (int, optional): batch size of training and test. Defaults to 128.
				data_dir (str, optional): path to the data directory.
				data_size (int, optional): size of training data. Defaults to 60000 is all data.
				pretrain_lr (float, optional): learning rate of pretraining DBN. Defaults to 0.1.
				pretrain_epochs (int, optional): number of training epochs of pretraining DBN. Defaults to 100.
		"""
		super().__init__()
		self.transform = transforms.Compose([
				transforms.ToTensor(),
				ThresholdTransform(thr=.5)
				])
		
		self.h_dims = h_dims
		self.pretrain = pretrain
		self.dbn: DBN = DBN(v_dim=num_features, h_dims=h_dims)
		self.dnn: DNN = None
		self.num_features = num_features
		self.num_classes = num_classes
		self.data_dir = data_dir
		self.data_size = data_size
		self.batch_size = batch_size
		self.lr = lr
		if self.pretrain:
			self.pretrain_lr = pretrain_lr
			self.pretrain_epochs = pretrain_epochs
			self.save_pretrain_path = save_pretrain_path
		self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
		self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
		self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
		if self.pretrain:
			self.save_hyperparameters(ignore=['num_features', 'num_classes', 'data_dir', 'save_pretrain_path', 'load_pretrain_path'])
		else:
			self.save_hyperparameters(ignore=['num_features', 'num_classes', 'data_dir', 'pretrain_lr', 'pretrain_epochs', 'save_pretrain_path', 'load_pretrain_path'])
		
		self.load_data()
		self.setup_pretrain(load_from=load_pretrain_path)
		
	def forward(self, x: torch.Tensor):
		return self.dnn(x)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr)
		return optimizer

	# before training with backprop, decide if perform pretraining
	def setup_pretrain(self, load_from=None) -> None:
		if self.pretrain:
			if load_from is not None:
				path = load_from + "dbn_pretrained_" + '_'.join([str(i) for i in self.h_dims]) +".pkl"
				try:
					with open(path, "rb") as f:
						self.dbn = pickle.load(f)
					print('load pretrained model')
				except:
					print(f'pretrained model not found in {path}, pretraining a new one')
					self._pretrain()
			else:
				self._pretrain()
		self.dnn = DNN(self.dbn, self.num_classes)
	
	def _pretrain(self):
		print('pretraining...')
		data = ThresholdTransform(thr=127)(self.mnist_train.data[:self.data_size])
		data = data.reshape(-1, self.num_features).numpy()
		self.dbn.fit(data=data, batch_size=self.batch_size, num_epochs=self.pretrain_epochs, lr=self.pretrain_lr)
		path = self.save_pretrain_path + "dbn_pretrained_" + '_'.join([str(i) for i in self.h_dims]) +".pkl"
		with open(path, "wb") as f:
				pickle.dump(self.dbn, f)
		print(f'pretrained model saved to {path}')


	def training_step(self, batch, batch_idx):
		x, y = batch
		y_pred_proba = self.dnn(x.view(-1, self.num_features))
		loss = F.nll_loss(y_pred_proba, y)
		y_pred = torch.argmax(y_pred_proba, dim=1)
		self.train_acc.update(y_pred, y)
		self.log_dict({
			'train_loss': loss, 
			'train_acc': self.train_acc}, 
			on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return loss
	
	# def validation_step(self, batch, batch_idx):
	# 	x, y = batch
	# 	y_pred_proba = self.model(x.view(-1, self.num_features))
	# 	loss = F.nll_loss(y_pred_proba, y)
	# 	y_pred = torch.argmax(y_pred_proba, dim=1)
	# 	self.val_acc.update(y_pred, y)

	# 	self.log_dict({
	# 		'val_loss': loss, 
	# 		'val_acc': self.val_acc},
	# 		on_step=False, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_pred_proba = self.dnn(x.view(-1, self.num_features))
		loss = F.nll_loss(y_pred_proba, y)
		y_pred = torch.argmax(y_pred_proba, dim=1)
		self.test_acc.update(y_pred, y)
		self.log_dict({
			'test_loss': loss, 
			'test_acc': self.test_acc},
			on_step=False, on_epoch=True, prog_bar=True, logger=True)

	def load_data(self) -> None:
		MNIST(self.data_dir, train=True, download=True)
		MNIST(self.data_dir, train=False, download=True)
		
		self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
		self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
	
	def train_dataloader(self):
		return DataLoader(Subset(self.mnist_train, list(range(self.data_size))), batch_size=self.batch_size, num_workers=8)


	def test_dataloader(self):
		return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=8)