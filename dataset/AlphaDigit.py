import torch
from torch.utils.data import Dataset, DataLoader
from utils import read_alpha_digit
import os

FILEPATH = os.path.dirname(__file__)+'/data/binaryalphadigs.mat'

class AlphaDigit(Dataset):
  def __init__(self, path=FILEPATH, label=range(36)) -> None:
    super().__init__()
    self.x, self.y = read_alpha_digit(path, label)
    self.x = torch.tensor(self.x, dtype=torch.float)
    self.y = torch.tensor(self.y, dtype=torch.float)
  def __len__(self):
      return len(self.y)
  def __getitem__(self, index):
      return self.x[index], self.y[index]