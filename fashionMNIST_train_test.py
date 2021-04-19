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
from sense import RandomProjection

from utils import IM_DIM, plot_train_results, plot_results, get_dataloaders

# Configuration
batch_size = 128
num_epochs = 20
# momentum = 0.9
lr = 0.01 # initial learning rate
lr_milestones = [8, 15]

val_split = 0.1  # proportion of training data to use for validation
bar_refresh_rate = 1  # how often to compute loss for display

# Crude way of determining if we're on CIS machine or laptop
n_workers = 32 if torch.cuda.is_available() else 0

compression_factors = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
test_accuracy = []

for i, cf in enumerate(compression_factors):
    trans = transforms.Compose([transforms.ToTensor(), RandomProjection(cf, IM_DIM)])

    # Dataset loading
    trainloader, valloader, testloader = get_dataloaders(batch_size, val_split, trans, n_workers)

    net = MNISTClassifier(resnet20(), lr, lr_milestones)

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=2, accelerator='ddp', max_epochs=num_epochs, progress_bar_refresh_rate=bar_refresh_rate)
    else:
        trainer = pl.Trainer(gpus=0, max_epochs=num_epochs, progress_bar_refresh_rate=bar_refresh_rate)

    trainer.fit(net, trainloader, valloader)
    trainer.test(model=net, test_dataloaders=testloader)

    test_accuracy.append(net.test_acc)
    # plot_train_results(net)

print(test_accuracy)
plot_results(compression_factors, test_accuracy)
