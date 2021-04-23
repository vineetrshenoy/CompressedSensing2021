# %% [code] {"jupyter":{"outputs_hidden":false}}
# Documentation for SPAMS
# http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams005.html#sec12
# !pip install spams

# %% [code] {"jupyter":{"outputs_hidden":false}}

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import time
from scipy.fftpack import dct

from math import log10, sqrt

import argparse
import cv2


import spams

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
# from sense.py
class CSTransform(object):

    def __init__(self, compression_factor, img_shp):

        self.rng = np.random.default_rng(seed=21) #Set RNG for repeatble results
        self.N =  img_shp[1] *img_shp[2] #length of vectorized image
        self.M = int(compression_factor * self.N)
        self.A = 0

    
    def __call__(self, tensor):

        shape = tensor.shape
        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector
        y = torch.matmul(self.A, x.type(torch.FloatTensor)) #get measurements
        img = torch.reshape(y, shape)
        
        return img

# from sense.py
class RandomProjection(CSTransform):


    def __init__(self, compression_factor, img_shp):
        
        super().__init__(compression_factor, img_shp)

        A = self.rng.standard_normal((int(self.M), self.N)) #sensing matrix
        A = np.transpose(scipy.linalg.orth(np.transpose(A)))
        self.A = torch.from_numpy(A).type(torch.FloatTensor) 


    def __call__(self, tensor):

        shape = tensor.shape
        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector
        y = torch.matmul(self.A, x.type(torch.FloatTensor)) #get measurements

        Atran = torch.transpose(self.A, 0, 1)

        proxy = torch.matmul(Atran, y) #proxy image
        proxy = torch.transpose(proxy, 0, 1)
        proxy = torch.reshape(proxy, shape)

        
        return proxy

# Inputs:
# ims: numpy array of the dataset
# S: the sparsity value to call in OMP, must be less than (28*28)
# A: the Torch matrix to use to compress the images
# Outputs:
# ims_compressed: the Torch matrix of compressed image data
# recovered_ims: the Torch dataset of sparsely recovered images
def speed_run_omp_on_batch(ims, S, A ):
    
    num_samples = ims.shape[0]
    M = A.shape[0]
    recovered_ims = np.empty((num_samples, 28, 28))
    D = dct(np.eye(28*28), axis=0)
    currA = ( np.matmul( A.numpy(), D)  )
    currA = currA/ np.tile(np.sqrt((currA*currA).sum(axis=0)),(currA.shape[0],1))
    currA = np.asfortranarray(currA)
    ims_compressed = np.empty((M, num_samples ))
    
    for i in range(num_samples):              

        temp = ims[i,:,:]
        temp = np.reshape(temp, (28*28))
        temp = np.matmul(A, temp) 
        ims_compressed[:,i]= temp
        
    ims_compressed = np.asfortranarray(ims_compressed)
    rec_all = spams.omp(X= ims_compressed, D=currA, L=S)
    
    for j in range(num_samples):
        rec = rec_all[:,j].toarray()
        rec = rec.flatten()
        rec = np.matmul(D,rec);
        recovered_ims[j,:,:] = np.reshape( rec ,(28,28))
        
    ims_compressed = torch.from_numpy(ims_compressed)
    recovered_ims = torch.from_numpy(recovered_ims)

    
    return ims_compressed, recovered_ims


def PSNR(original, compressed): #credits to https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  
        return np.NAN
    max_pixel = 255.0
    psnr = 10 * log10(max_pixel / sqrt(mse))
    return psnr

# Inputs
# og: original MNIST dataset in numpy
# rec: recovered dataset in numpy
# Outputs
# psnr_vals: a numpy array of psnr values corresponding to reconstructed images
def compute_psnr_on_datasets(og, rec):
    psnr_vals = np.empty((og.shape[0]))
    for i in range(og.shape[0]):
        im1 = np.squeeze(og[i,:,:])
        im2 = np.squeeze(rec[i,:,:])
        psnr_vals[i] = PSNR(im1,im2)
        
    return psnr_vals

# %% [code]

# GOAL: transform ENTIRE fashion MNIST into sparse images
# substitute the data for the recovered image version
# how, you ask?
# just sub in a numpy array 
# then switch to # dtype=torch.uint8 which is default type for torch.from_numpy()

# %% [code]
# Define constants ahead of time

S = 50
cf = 0.5
tfobj = RandomProjection(cf, (1, 28, 28) )
A = tfobj.A

trainset_full = FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor()) 
# YOU DON'T NEED THE NEW TRANSFORM IN THE DATASET BECAUSE YOU HAVE ALREADY DONE THE TRANSFORMING!

ims = trainset_full.data.numpy()
init = time.time()
num_ims = 100 # number of samples for testing
[compressed, recovered] = speed_run_omp_on_batch(ims[:num_ims,:,:], S,A)
new_trainset_full = trainset_full
new_trainset_full.data = recovered #rewrite just the label object and leave the labels


end = time.time()
print( (end-init) )

# %% [code]
psnr_recovered = compute_psnr_on_datasets(ims[:num_ims,:,:],recovered.numpy())
print("Average PSNR:")
print(np.nanmean(psnr_recovered))
print("# of perfect recovery")
print(np.sum(np.isnan(psnr_recovered)))

# %% [code]
# print an image before and after recovery
plt.imshow(np.squeeze(ims[0,:,:])), plt.title("Original")
npset = new_trainset_full.data.numpy()
plt.figure()
plt.imshow(np.squeeze(npset[0,:,:])), plt.title("After Sparse Recovery")
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
