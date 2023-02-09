# import os
# import math
import torch
from torch import optim
from .base import BaseVAE
from .types_ import *
import pytorch_lightning as pl
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
# from torchvision import transforms
# import torchvision.utils as vutils
# from torch.utils.data import DataLoader

class CVAELit(pl.LightningModule):
	def __init__(self,
	      			 model: BaseVAE,
							 ):
		super().__init__()
		self.model = model
	
	def training_step(self, batch, batch_idx):
		x, label = batch
		# print(x, label)
		y = self.onehot_encoder(label)
		batch_size = x.size(0)
		# keep batch dim, flatten the rest
		x = x.view(batch_size, -1) 

		x_re, mu, logsd = self.model.forward(x, y)
		loss, recons_loss, kld_loss = self.model.loss_function(x, x_re, mu, logsd)
		# assert kld_loss.isnan().any() == False
		# assert recons_loss.isnan().any() == False
		# assert loss.isnan().any() == False

		self.log_dict(
				{'train_loss': loss / batch_size, 
					'reconstruction_loss': recons_loss / batch_size,
					'KL-divergence': kld_loss/ batch_size
					},
					prog_bar=True,
					logger=True,
					on_step=True,
					on_epoch=True,
			)
		return loss

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=1e-3)
		return optimizer
	
	def onehot_encoder(self, label: Tensor, nb_digits=10):
		# take labels (from the dataloader) and return labels onehot-encoded
		label_onehot = torch.zeros(size=(label.shape[0], nb_digits), device = label.device)
		label_onehot.zero_()
		label_onehot.scatter_(1, label.view(-1,1), 1)
		# label_onehot = label_onehot.to(device)
		return label_onehot