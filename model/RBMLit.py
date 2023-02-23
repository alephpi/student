from typing import Tuple
import pytorch_lightning as pl
import torch
from torch import Tensor
from .RBM import RBM
import torch.nn.functional as F
class RBMLit(pl.LightningModule):
  def __init__(self, v_dim: int, h_dim: int, k=1, lr=1e-1) -> None:
    super().__init__()
    self.num_features = v_dim
    self.lr = lr
    self.model = RBM(v_dim=v_dim, h_dim=h_dim, k=1)

  def forward(self, v:Tensor) -> Tuple[Tensor, Tensor]:
    return self.model(v)

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
    return optimizer

  def training_step(self, training_batch, batch_idx):
    x, _ = training_batch
    v, v_k = self.model.forward(x.view(-1, self.num_features))
    # Contrastive divergence is the approximation of gradient, which corresponds to the following surrogate loss
    # backprop is taken care by the auto-differentiation of torch and implicitly taken care by the pytorch-lightining framework
    loss = self.model.free_energy(v) - self.model.free_energy(v_k)
    l2_loss = F.mse_loss(v, v_k)
    self.log_dict(
      {'train loss':loss,
       'l2 loss': l2_loss},
       prog_bar=True,
       logger=True,
       on_step=True,
       on_epoch=True)
    return loss

  @torch.no_grad()
  def inference(self, v, n_gibbs):
    '''
    n_gibbs: iteration times of gibbs sampling
    '''
    v_ = v
    for i in range(n_gibbs):
      v_ = self.model.gibbs_sampling(v_)
    return v_
