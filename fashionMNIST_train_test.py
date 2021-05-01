# -*- coding: utf-8 -*-
"""
This script trains and tests ResNet-20 on Fashion-MNIST

Author(s): Arun Reddy
"""

import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from resnet import resnet20
from classifiers import MNISTClassifier
from sense import RandomProjection, RSTD, USTD, RSFD, LFS, EFS
import numpy as np

from utils import IM_DIM, plot_train_results, plot_results, get_dataloaders, get_sparse_recovered_dataloaders

# Configuration
batch_size = 128
num_epochs = 20
lr = 0.01  # initial learning rate
lr_milestones = [8, 15]

val_split = 0.1  # proportion of training data to use for validation
bar_refresh_rate = 1  # how often to compute loss for display

# Crude way of determining if we're on CIS machine or laptop
n_workers = 32 if torch.cuda.is_available() else 0

compression_factors = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
sensing_schemes = [RandomProjection, RSTD]
scheme_names = ["Gaussian Sensing", "Random Subsampling"]
S = 220
test_accuracy = np.zeros((len(sensing_schemes), len(compression_factors)))

# Loop over sensing schemes and compression factors
for i, ss in enumerate(sensing_schemes):
    for j, cf in enumerate(compression_factors):
        # Define the data transformation for this network
        sensing_transform = ss(cf, IM_DIM)
        trans = transforms.Compose([transforms.ToTensor(), sensing_transform])

        # Build the dataloaders
        # trainloader, valloader, testloader = get_dataloaders(batch_size, val_split, trans, n_workers)
        trainloader, valloader, testloader = get_sparse_recovered_dataloaders(sensing_transform, S, batch_size, val_split, n_workers)
        # Construct the model
        net = MNISTClassifier(resnet20(), lr, lr_milestones)

        if torch.cuda.is_available():
            trainer = pl.Trainer(gpus=2, accelerator='ddp', max_epochs=num_epochs, progress_bar_refresh_rate=bar_refresh_rate)
        else:
            trainer = pl.Trainer(gpus=0, max_epochs=num_epochs, progress_bar_refresh_rate=bar_refresh_rate)

        # Train the network
        trainer.fit(net, trainloader, valloader)
        # Test the network
        trainer.test(model=net, test_dataloaders=testloader)
        # Save off the test accuracy for this network
        test_accuracy[i][j] = net.test_acc

        # plot_train_results(net)

print(test_accuracy)
plot_results(compression_factors, test_accuracy, scheme_names)
