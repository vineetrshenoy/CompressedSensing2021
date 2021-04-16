import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import cv2

class RandomProjection(object):


    def __init__(self, compression_factor, img_shp):
        
        self.rng = np.random.default_rng(seed=21) #Set RNG for repeatble results
        N =  img_shp[1] *img_shp[2] #length of vectorized image
        M = compression_factor * N

        A = self.rng.standard_normal((int(M), N)) #sensing matrix
        A = np.transpose(scipy.linalg.orth(np.transpose(A)))
        self.A = torch.from_numpy(A) 


    def __call__(self, tensor):

        shape = tensor.shape
        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector
        y = torch.matmul(self.A, x.type(torch.DoubleTensor)) #get measurements

        Atran = torch.transpose(self.A, 0, 1)

        proxy = torch.matmul(Atran, y) #proxy image
        proxy = torch.transpose(proxy, 0, 1)
        proxy = torch.reshape(proxy, shape)

        
        return proxy


if __name__ == '__main__':

    # Read image
    img = cv2.imread("./proxy_image/test_img.jpeg", cv2.IMREAD_GRAYSCALE)

    # Resize the image for faster processing / less memory
    resize_factor = 0.1
    im_dim = np.shape(img)
    img = cv2.resize(img, (int(im_dim[1]*resize_factor), int(im_dim[0]*resize_factor)))

    im_dim = np.shape(img)[0:2]
    N = im_dim[0]*im_dim[1]
    
    compression_factor = 0.1
    '''
    # Vectorize the image
    img_vec = np.reshape(img, (N, 1))

    # Compressively sense the image
    compression_factor = 0.1
    M = int(np.floor(compression_factor * N))

    A_gauss = (np.random.randn(M, N))
    # A_gauss = np.transpose(scipy.linalg.orth(np.transpose(A_gauss)))
    # print(np.shape(A_gauss))
    y_gauss = np.matmul(A_gauss, img_vec)

    y_img = np.reshape(np.matmul(np.transpose(A_gauss), y_gauss), (im_dim[0], im_dim[1]))
    '''

    timg = torch.from_numpy(img)
    timg = torch.unsqueeze(timg, 0)

    RP = RandomProjection(compression_factor, timg.shape)
    y_img = RP(timg)
    y_img = y_img[0, :].numpy()

    np.save('trans.npy', y_img)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(y_img, cmap='gray')
    plt.title("Proxy")
    plt.savefig('test.jpg')



