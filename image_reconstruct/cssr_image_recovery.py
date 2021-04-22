# %% [code] {"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false}}

# credits go to:
# https://github.com/rfmiotto/CoSaMP/blob/master/cosamp.ipynb
def cosamp(Phi, u, s, tol=1e-10, max_iter=1000):
    """
    @Brief:  "CoSaMP: Iterative signal recovery from incomplete and inaccurate
             samples" by Deanna Needell & Joel Tropp

    @Input:  Phi - Sampling matrix
             u   - Noisy sample vector
             s   - Sparsity vector

    @Return: A s-sparse approximation "a" of the target signal
    """
    max_iter -= 1 # Correct the while loop
    num_precision = 1e-12
    a = np.zeros(Phi.shape[1])
    v = u
    iter = 0
    halt = False
    while not halt:
        iter += 1
#         print("Iteration {}\r".format(iter))
        
        y = abs(np.dot(np.transpose(Phi), v))
        Omega = [i for (i, val) in enumerate(y) if val > np.sort(y)[::-1][2*s] and val > num_precision] # quivalent to below
        #Omega = np.argwhere(y >= np.sort(y)[::-1][2*s] and y > num_precision)
        T = np.union1d(Omega, a.nonzero()[0])
        #T = np.union1d(Omega, T)
        b = np.dot( np.linalg.pinv(Phi[:,T]), u )
        igood = (abs(b) > np.sort(abs(b))[::-1][s]) & (abs(b) > num_precision)
        T = T[igood]
        a[T] = b[igood]
        v = u - np.dot(Phi[:,T], b[igood])
        
        halt = np.linalg.norm(v)/np.linalg.norm(u) < tol or \
               iter > max_iter
        
    return a


# %% [code]
def produce_gaussian_sampling(compression_factor, img_shp): #based of Vineet's RandomProjection object in sense.py
        rng1 = np.random.default_rng(seed=21) #Set RNG for repeatble results
        N =  img_shp[0] *img_shp[1] #length of vectorized image
        M = int(compression_factor * N)

        A = rng1.standard_normal(( (M), N)) #sensing matrix
        A = np.transpose(scipy.linalg.orth(np.transpose(A)))
        return A, M

# %% [code]
def run_cosamp_on_batch(ims, c_factor, s_val ):
    num_samples = ims.shape[0]
    ims_compressed = np.empty((num_samples, int(c_factor*28*28) ))
    recovered_ims = np.empty((num_samples, 28, 28))
    for i in range(num_samples):        
        A, M = produce_gaussian_sampling(c_factor, (28,28))
        temp = ims[i,:,:]
        temp = np.reshape(temp, (28*28))
        ims_compressed[i,:]= np.matmul(A, temp)
        
        
#         rec = cosamp(A, ims_compressed[i,:], s_val) 
#         recovered_ims[i,:,:] = np.reshape( rec ,(28,28))

    
    return ims_compressed, recovered_ims

# %% [code]
## ALSO NEED TO ADD SOME WAY OF CHECKING ACCURACY LIKE THEIR MODEL DOES 

# %% [code]
trainset_full = torchvision.datasets.FashionMNIST(root="data", train=True,
                                             download=True, transform=transforms.ToTensor())

# %% [code] {"jupyter":{"outputs_hidden":false}}
ims = trainset_full.data.numpy()
labs = trainset_full.targets.numpy()
# ims_compressed = np.empty((60,m_val))
# for i in range(60): # np.size(ims,0)
#     temp = np.reshape(ims[i,:,:],(28*28))
#     ims_compressed[i,:]= np.matmul(A, temp)
#     y = ims_compressed[0,:]
#     recovered = cosamp(A,y, 20) 
#     print(np.size(idk))

# %% [code]
init = time.time()
run_cosamp_on_batch(ims[:20,:,:], 0.5, 40)
end = time.time()
print( (end-init)/60 )

# SCRATCH WORK:

# %% [code] {"jupyter":{"outputs_hidden":false}}
ims_compressed.shape

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
test_im = np.reshape(idk,(28,28))
plt.imshow(test_im)

# %% [code] {"jupyter":{"outputs_hidden":false}}
