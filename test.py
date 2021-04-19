import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from resnet import resnet20
from classifiers import MNISTClassifier
import numpy as np
from utils import plot_train_results

# Path to the checkpoint you want to use for testing
ckpt_path = "lightning_logs/version_10/checkpoints/epoch=19-step=4219.ckpt"

# Crude way of determining if we're on CIS machine or laptop
n_workers = 32 if torch.cuda.is_available() else 0

testset = torchvision.datasets.FashionMNIST(root="data", train=False,
                                       download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=n_workers)

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

net = MNISTClassifier(resnet20())
net.load_state_dict(ckpt['state_dict'])

if torch.cuda.is_available():
    trainer = pl.Trainer(gpus=-1, accelerator='ddp')
else:
    trainer = pl.Trainer(gpus=0)

trainer.test(model=net, test_dataloaders=testloader)
plot_train_results(net)


