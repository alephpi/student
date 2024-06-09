# CVAE model
# code template based on AntixK/PyTorch-VAE
# network architecture based on cdoersch/vae_tutorial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .types_ import *
from .base import BaseVAE
class CVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 30,
                 hidden_dims: List[int] = [1000, 500, 250],
                 n_classes: int = 10
                 ):
        super().__init__()

        self.latent_dim = latent_dim

        encoder_layers = [in_channels]
        encoder_layers.extend(hidden_dims)

        # Encoder
        modules = []
        for i in range(len(encoder_layers)-2):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=encoder_layers[i],
                              out_features=encoder_layers[i+1]),
                    nn.LeakyReLU()
                ))
        modules.append(nn.Linear(in_features=encoder_layers[-2], out_features=encoder_layers[-1]))
        # last layer is dealt independently since it uses relu for mu but not for logsd 

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(in_features = encoder_layers[-1], out_features=latent_dim)
        self.fc_logsd = nn.Linear(in_features = encoder_layers[-1], out_features=latent_dim)

        # Decoder
        modules = []
        # the first layer of decoder is with addition of a class label input
        decoder_layers = [latent_dim + n_classes]
        decoder_layers.extend(hidden_dims[::-1])
        for i in range(len(decoder_layers)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=decoder_layers[i],
                              out_features=decoder_layers[i+1]),
                    nn.LeakyReLU() 
                ))
        modules.append(nn.Linear(in_features=decoder_layers[-1], out_features=in_channels))
        self.decoder = nn.Sequential(*modules)

    def encode(self, x: Tensor) -> Tuple[Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(F.leaky_relu(h))
        logsd  = self.fc_logsd(h)
        return mu, logsd # mean & log standard deviation of q(z|x)
    
    def reparameterize(self, mu: Tensor, logsd: Tensor) -> Tensor:
        std = torch.exp(logsd)
        eps = torch.randn_like(std) # generate standard gaussian variable
        return mu + eps * std # make backpropa possible

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder(z)
        x_re = torch.sigmoid(h)
        return x_re

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor]:
        # assert x.isnan().any() == False, print(x)
        mu, logsd = self.encode(x)
        # assert mu.isnan().any() == False, print(mu)
        # assert logsd.isnan().any() == False, print(logsd)
        z = self.reparameterize(mu, logsd)
        # assert z.isnan().any() == False, print(z)
        x_re = self.decode(torch.cat([z,y], dim=1)) 
        # assert x_re.isnan().any() == False, print(x_re)
        return x_re, mu, logsd
    
    def loss_function(self, x: Tensor, x_re: Tensor, mu: Tensor, logsd: Tensor, kld_weight: int =1, **kwargs) -> Tuple[Tensor]:
        # recons_loss = F.mse_loss(x, x_re)
        # why reduction is sum not mean?
        # bce is not symmetric, first should be a distribution and second is one-hot.
        recons_loss = F.binary_cross_entropy(x_re, x, reduction='sum')
        # recons_loss.retain_grad()
        # assert recons_loss.isnan().any() == False, print(recons_loss)
        logvar = 2 * logsd
        # assert (logvar.exp()).isinf().any() == False, print('logvar.exp nan')
        kld_loss = - 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        # kld_loss.retain_grad()
        # assert kld_loss.isnan().any() == False, print(kld_loss)
        loss = recons_loss + kld_weight * kld_loss
        return loss, recons_loss, kld_loss

    # the following methods are provided for generation and visualization 
    @torch.no_grad()
    def sample(self,
               num_samples: int,
               device: torch.device) -> Tensor:
        '''
        sample from latent space and decode to image
        '''
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    @torch.no_grad()
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        '''
        reconstruct x from model
        '''
        return self.forward(x)[0]