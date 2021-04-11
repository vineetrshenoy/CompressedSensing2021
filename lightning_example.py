# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 23:23:59 2020

@author: reddyav1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, trange
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# from densenet import densenet121
from resnet import resnet20

# Configuration
batch_size = 128
num_epochs = 50
momentum = 0.9
learning_rate = 0.001
test_batch_size = 10000
val_split = 0.1  # proportion of training data to use for validation

use_saved_model = True
load_filename = "reddynetv1"
save_model = True
save_filename = "reddynetv1"

bar_refresh_rate = 1  # how often to compute loss for display


class ReddyNetv1(nn.Module):
    def __init__(self):
        super().__init__()
        # Layers
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(2 * 2 * 128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 2 * 2 * 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # The actual neural network
        self.backbone = backbone

        # Things that need to be saved across training session
        self.training_losses = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.overall_accuracy = None

    def forward(self, x):
        x = self.backbone(x)
        return x

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=0.001) # use for ReddyNet

        optimizer = torch.optim.SGD(self.parameters(), lr=0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[15, 30], last_epoch=0 - 1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.training_losses.append(loss.cpu().detach().numpy())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.validation_losses.append(loss.cpu().detach().numpy())
        _, predicted_class = torch.max(y_hat, 1)
        accuracy = torch.sum(predicted_class == y).cpu().detach().numpy() / y.size(0)
        self.validation_accuracies.append(accuracy)
        self.log('val_acc', accuracy, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        _, predicted_class = torch.max(y_hat, 1)
        # Convert to numpy
        y = y.cpu().detach().numpy()
        predicted_class = predicted_class.cpu().detach().numpy()
        # Compute accuracy
        accuracy = np.sum(predicted_class == y) / len(y)
        self.overall_accuracy = accuracy
        print('Overall Accuracy: %.3f' % accuracy)
        # Create a confusion matrix
        cm_normalized = confusion_matrix(y, predicted_class, normalize='true')
        self.cm = cm_normalized


def plot_results(model):
    if not (os.path.isdir("outputs")):
        os.mkdir("outputs")
    # Plot validation accuracy and loss on a single plot
    plt.rcParams.update({'font.size': 10})
    fig, (ax1, ax3) = plt.subplots(2, 1)
    x_val = np.arange(len(model.validation_accuracies))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val. Loss', color=color)
    ax1.plot(x_val, np.array(model.validation_losses), color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val. Accuracy', color=color)
    ax2.plot(x_val, model.validation_accuracies, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.grid(True)

    # Plot training losses in another subplot
    x = np.linspace(0, len(model.validation_accuracies) - 1, len(model.training_losses))
    color = 'tab:red'
    ax3.plot(x, model.training_losses, color=color)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Loss')
    ax3.grid(True)

    fig.tight_layout()
    plt.savefig("outputs/training_plot.png", dpi=500)
    plt.show()

    # Plot confusion matrix
    plt.figure()
    plt.rcParams.update({'font.size': 5})
    cm_plot = ConfusionMatrixDisplay(confusion_matrix=model.cm, display_labels=classes)
    cm_plot.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Total Accuracy = %.3f%%)' % (100 * model.overall_accuracy))
    plt.savefig("outputs/confusion_matrix.png", dpi=500)


# Dataset loading

train_transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

trainset_full = torchvision.datasets.CIFAR10(root="data", train=True,
                                             download=True, transform=train_transform)

trainset, valset = torch.utils.data.random_split(trainset_full, [int((1 - val_split) * len(trainset_full)),
                                                                 int(val_split * len(trainset_full))])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset),
                                        shuffle=False, num_workers=0)

testset = torchvision.datasets.CIFAR10(root="data", train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net = CIFAR10Classifier(ReddyNetv1())
net = CIFAR10Classifier(resnet20())

if torch.cuda.is_available():
    trainer = pl.Trainer(gpus=-1, max_epochs=num_epochs, progress_bar_refresh_rate=bar_refresh_rate)
else:
    trainer = pl.Trainer(gpus=0, max_epochs=num_epochs, progress_bar_refresh_rate=bar_refresh_rate)

trainer.fit(net, trainloader, valloader)
trainer.test(model=net, test_dataloaders=testloader)

plot_results(net)