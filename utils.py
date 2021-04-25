import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader
from math import log10, sqrt
import spams
import seaborn as sns
from scipy.fftpack import dct
sns.set_theme()

DPI = 300  # dpi for saving figures

# MNIST class names
# classes = tuple([str(x) for x in range(10)])

# Fashion MNIST class names
classes = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot')

IM_DIM = (1, 28, 28)  # shape of MNIST images
N = IM_DIM[1]*IM_DIM[2]


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

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.grid(False)
    plt.show()

# very similar to get_dataloaders
# New Inputs:
# trans: the object outputted by a function like RandomProjection
# S: the sparsity value to call in OMP, must be less than (28*28)
# New Output:
# psnr_recovered: the array of psnr recovered corresponding to each recovered image
def get_sparse_recovered_dataloaders(trans, S, batch_size, val_split,  n_workers):
    A = trans.A
    ts_full = FashionMNIST(root="data", train=True, download=True,  transform=transforms.ToTensor())
    testset = FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
    trainset_full = ts_full
    ims = ts_full.data.numpy()
    ims_test = testset.data.numpy()

    # init = time.time()
    [compressed, recovered] = speed_run_omp_on_batch(ims, S, A)
    trainset_full.data = recovered
    [compressed, recovered] = speed_run_omp_on_batch(ims_test, S, A)
    testset.data = recovered
    # end = time.time()

    # print("Time of Sparse Recovery (s):")
    # print( (end-init) )
    # psnr_recovered = compute_psnr_on_datasets(ims,recovered.numpy())

    trainset, valset = random_split(trainset_full, [int((1 - val_split) * len(trainset_full)), int(val_split * len(trainset_full))])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    valloader = DataLoader(valset, batch_size=len(valset), shuffle=False, num_workers=n_workers)


    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=n_workers)

    return trainloader, valloader, testloader

def get_dataloaders(batch_size, val_split, transforms, n_workers):
    trainset_full = FashionMNIST(root="data", train=True, download=True, transform=transforms)
    trainset, valset = random_split(trainset_full, [int((1 - val_split) * len(trainset_full)), int(val_split * len(trainset_full))])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    valloader = DataLoader(valset, batch_size=len(valset), shuffle=False, num_workers=n_workers)

    testset = FashionMNIST(root="data", train=False, download=True, transform=transforms)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=n_workers)

    return trainloader, valloader, testloader


def plot_results(compression_factors, test_accuracies, scheme_names):
    for n in range(test_accuracies.shape[0]):
        make_cf_barplot(compression_factors, test_accuracies[n, :], scheme_names[n])


def make_cf_barplot(compression_factors, test_accuracy, scheme_name):
    plt.figure()
    x_vals = ["{cf:.0f}%\n(M={M})".format(cf=cf * 100, M=int(N*cf)) for cf in compression_factors]
    splot = sns.barplot(x=x_vals, y=[test_acc * 100 for test_acc in test_accuracy])
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.1f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')
    plt.ylim(0, 100)
    plt.xlabel("Compression Factor\n(Measurement Size)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy by Compression Factor\n({ss})".format(ss=scheme_name))
    plt.tight_layout()

    # Save figure
    if not (os.path.isdir("outputs")):
        os.mkdir("outputs")
    fig_path = "outputs/accuracy_across_cf_{ss}.png".format(ss=scheme_name.replace(' ', '_'))
    plt.savefig(fig_path, dpi=DPI)

    plt.show()


def plot_train_results(model):
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
    plt.savefig("outputs/training_plot.png", dpi=DPI)
    plt.show()

    # Plot confusion matrix
    plt.figure()
    plt.rcParams.update({'font.size': 5})
    cm_plot = ConfusionMatrixDisplay(confusion_matrix=model.cm, display_labels=classes)
    cm_plot.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Total Accuracy = %.3f%%)' % (100 * model.test_acc))
    plt.savefig("outputs/confusion_matrix.png", dpi=DPI)