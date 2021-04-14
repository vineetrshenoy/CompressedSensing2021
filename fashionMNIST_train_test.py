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

from utils import plot_results

# Configuration
batch_size = 128
num_epochs = 20
momentum = 0.9
learning_rate = 0.001
val_split = 0.1  # proportion of training data to use for validation

bar_refresh_rate = 1  # how often to compute loss for display

# Crude way of determining if we're on CIS machine or laptop
n_workers = 32 if torch.cuda.is_available() else 0

# Dataset loading
trainset_full = torchvision.datasets.FashionMNIST(root="data", train=True,
                                             download=True, transform=transforms.ToTensor())

trainset, valset = torch.utils.data.random_split(trainset_full, [int((1 - val_split) * len(trainset_full)),
                                                                 int(val_split * len(trainset_full))])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset),
                                        shuffle=False, num_workers=n_workers)

testset = torchvision.datasets.FashionMNIST(root="data", train=False,
                                       download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=n_workers)

net = MNISTClassifier(resnet20())

if torch.cuda.is_available():
    trainer = pl.Trainer(gpus=-1, accelerator='ddp', max_epochs=num_epochs, progress_bar_refresh_rate=bar_refresh_rate)
else:
    trainer = pl.Trainer(gpus=0, max_epochs=num_epochs, progress_bar_refresh_rate=bar_refresh_rate)

trainer.fit(net, trainloader, valloader)
trainer.test(model=net, test_dataloaders=testloader)

plot_results(net)
