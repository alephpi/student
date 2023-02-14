import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
import arff
import random

from src.utils import MyDataset

def get_bibtex(dir_path: str, use_train: bool):
    """
    Load the bibtex dataset.
    __author__ = "Michael Gygli, ETH Zurich"
    from https://github.com/gyglim/dvn/blob/master/mlc_datasets/__init__.py
    number of labels ("tags") = 159
    dimension of inputs = 1836
    Returns
    -------
    txt_labels (list)
        the 159 tags, e.g. 'TAG_system', 'TAG_social_nets'
    txt_inputs (list)
        the 1836 attribute words, e.g. 'dependent', 'always'
    labels (np.array)
        N x 159 array in one hot vector format
    inputs (np.array)
        N x 1839 array in one hot vector format
    """
    feature_idx = 1836
    if use_train:
        dataset = arff.load(open(os.path.join(dir_path, 'bibtex-train.arff')), "r")
    else:
        dataset = arff.load(open(os.path.join(dir_path, 'bibtex-test.arff')), "r")

    data = np.array(dataset['data'], np.int)

    labels = data[:, feature_idx:]
    inputs = data[:, 0:feature_idx]
    txt_labels = [t[0] for t in dataset['attributes'][feature_idx:]]
    txt_inputs = [t[0] for t in dataset['attributes'][:feature_idx]]
    return labels, inputs, txt_labels, txt_inputs


def normalize_inputs(inputs):
    
    mean = np.mean(inputs, axis=0).reshape((1, -1))
    std = np.std(inputs, axis=0).reshape((1, -1)) + 10 ** -6

    train_inputs = inputs.astype(float)
    train_inputs -= mean
    train_inputs /= std
    return train_inputs


def load_data_loader(X, Y, use_cuda: bool,
                     batch_size=32,norm_inputs=True,
                     shuffle=True):

    print('Preparing the data loader...')

    if norm_inputs:
        X = normalize_inputs(X)
    data = MyDataset(X, Y)
    
    n = X.shape[0]
    indices = list(range(n))

    if shuffle:
        random.shuffle(indices)

    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices),
        pin_memory=use_cuda
    )

    return data_loader
