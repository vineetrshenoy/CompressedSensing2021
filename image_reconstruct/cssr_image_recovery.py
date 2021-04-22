# %% [code] {"jupyter":{"outputs_hidden":false}}
# Documentation for SPAMS
# http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams005.html#sec12
!pip install spams

# %% [code] {"jupyter":{"outputs_hidden":false}}

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
from scipy.fftpack import dct

import spams

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false}}
trainset_full = torchvision.datasets.FashionMNIST(root="data", train=True,
                                             download=True, transform=transforms.ToTensor())

# %% [code] {"jupyter":{"outputs_hidden":false}}
def produce_gaussian_sampling(compression_factor, img_shp): #based off Vineet's RandomProjection object in sense.py
        rng1 = np.random.default_rng(seed=21) #Set RNG for repeatble results
        N =  img_shp[0] *img_shp[1] #length of vectorized image
        M = int(compression_factor * N)

        A = rng1.standard_normal(( (M), N)) #sensing matrix
        A = np.transpose(scipy.linalg.orth(np.transpose(A)))
        return A, M

# %% [code]
def speed_run_omp_on_batch(ims, c_factor, S ):
    num_samples = ims.shape[0]
    
    recovered_ims = np.empty((num_samples, 28, 28))
    D = dct(np.eye(28*28), axis=0)
    A, M = produce_gaussian_sampling(c_factor, (28,28))
    currA = ( np.matmul(A,D)  )
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

    
    return ims_compressed, recovered_ims

# %% [code] {"jupyter":{"outputs_hidden":false}}
## ALSO NEED TO ADD SOME WAY OF CHECKING ACCURACY LIKE THEIR MODEL DOES

# %% [code] {"jupyter":{"outputs_hidden":false},"_kg_hide-output":true,"_kg_hide-input":true}


# %% [code] {"jupyter":{"outputs_hidden":false}}
ims = trainset_full.data.numpy()
labs = trainset_full.targets.numpy()

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code]
init = time.time()
[compressed, recovered] = speed_run_omp_on_batch(ims[:1000,:,:], .5, 50)
end = time.time()
print( (end-init) )

# %% [code]
test_im = np.reshape(recovered[1,:,:],(28,28))
plt.imshow(test_im)

# %% [code] {"jupyter":{"outputs_hidden":false}}
test_im = np.reshape(recovered[0,:,:],(28,28))
plt.imshow(test_im)

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
