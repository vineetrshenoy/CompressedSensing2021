import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import cv2

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

class RandomProjection(object):


    def __init__(self, compression_factor, img_shp):
        
        self.rng = np.random.default_rng(seed=21) #Set RNG for repeatble results
        N =  img_shp[1] *img_shp[2] #length of vectorized image
        M = compression_factor * N

        A = self.rng.standard_normal((int(M), N)) #sensing matrix
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


class RSTD(CSTransform): #Random Sampling Time Domain

    def __init__(self, compression_factor, img_shp):
 
        super().__init__(compression_factor, img_shp)
        
        
        self.A = np.eye(self.N)
        all_idx = np.arange(0, self.N, 1)
        idx = self.rng.choice(self.N, self.M, replace=False)
        dif = np.setdiff1d(all_idx, idx)
        self.A[dif, :] = 0
        self.A = torch.from_numpy(self.A).type(torch.FloatTensor)

        temp = np.eye(self.N)
        self.temp = temp[idx, :]
        self.temp = torch.from_numpy(self.temp).type(torch.FloatTensor)
        '''
        ort = np.transpose(scipy.linalg.orth(np.transpose(self.temp)))

        #returns false, locations of nonzero are same, but not values
        flag = np.array_equal(self.temp, ort) 
        
        temp_nz = np.nonzero(self.temp)
        ort_nz = np.nonzero(ort)

        print(self.temp[temp_nz[0][0], temp_nz[1][0]])
        print(ort[ort_nz[0][0], ort_nz[1][0]])

        
        '''
    def __call__(self, tensor):

        shape = tensor.shape
        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector
        y = torch.matmul(self.A, x.type(torch.FloatTensor)) #get measurements
        y_temp = torch.matmul(self.temp, x.type(torch.FloatTensor))
        
        img_temp =  torch.matmul(torch.transpose(self.temp, 0, 1), y_temp) 
        img_temp = torch.reshape(img_temp, shape)
        img = torch.reshape(y, shape)

        return img_temp
    
    
    
    '''
    def __call__(self, tensor):

        shape = tensor.shape
        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector
        y = torch.matmul(self.A, x.type(torch.FloatTensor)) #get measurements
        y_temp = torch.matmul(self.temp, x.type(torch.FloatTensor))
        
        img_temp =  torch.matmul(torch.transpose(self.temp, 0, 1), y_temp) 
        img_temp = torch.reshape(img_temp, shape)
        img = torch.reshape(y, shape)

        find_dif = torch.eq(img, img_temp)
        print(find_dif.sum()) #same length as img and img_temp, exactly the same
        
        return img
    '''


class USTD(CSTransform): #Uniform Sampling Time Domain

    def __init__(self, compression_factor, img_shp):
 
        super().__init__(compression_factor, img_shp)
        
        
        self.A = np.eye(self.N)
        step = np.floor(self.N / self.M)
        all_idx = np.arange(0, self.N, 1)
        idx = np.arange(0, self.N, step.astype(int))
        dif = np.setdiff1d(all_idx, idx)
        self.A[dif, :] = 0
        self.A = torch.from_numpy(self.A).type(torch.FloatTensor)


class RSFD(CSTransform): #Random Sampling Frequency Domain

    def __init__(self, compression_factor, img_shp):
 
        super().__init__(compression_factor, img_shp)
        
        
        self.A = scipy.fft.dct(np.eye(self.N))
        all_idx = np.arange(0, self.N, 1)
        idx = self.rng.choice(self.N, self.M, replace=False)
        dif = np.setdiff1d(all_idx, idx)
        self.A[dif, :] = 0
        self.A = torch.from_numpy(self.A).type(torch.FloatTensor)


class LFS(CSTransform): #Low Frequency Sampling

    def __init__(self, compression_factor, img_shp):
 
        super().__init__(compression_factor, img_shp)
        
        
        self.A = scipy.fft.dct(np.eye(self.N))
        all_idx = np.arange(0, self.N, 1)
        idx = np.arange(0, self.M, 1)
        dif = np.setdiff1d(all_idx, idx)
        self.A[dif, :] = 0
        self.A = torch.from_numpy(self.A).type(torch.FloatTensor)



class EFS(CSTransform): #Equispaced frequency sampling

    def __init__(self, compression_factor, img_shp):
 
        super().__init__(compression_factor, img_shp)
        
        
        self.A = scipy.fft.dct(np.eye(self.N))
        step = np.floor(self.N / self.M)
        all_idx = np.arange(0, self.N, 1)
        idx = np.arange(0, self.N, step.astype(int))
        dif = np.setdiff1d(all_idx, idx)
        self.A[dif, :] = 0
        self.A = torch.from_numpy(self.A).type(torch.FloatTensor)



if __name__ == '__main__':

    # Read image
    img = cv2.imread("./proxy_image/test_img.jpeg", cv2.IMREAD_GRAYSCALE)

    # Resize the image for faster processing / less memory
    resize_factor = 0.1
    im_dim = np.shape(img)
    img = cv2.resize(img, (int(im_dim[1]*resize_factor), int(im_dim[0]*resize_factor)))

    im_dim = np.shape(img)[0:2]
    N = im_dim[0]*im_dim[1]
    
    compression_factor = 0.5
    

    timg = torch.from_numpy(img)
    timg = torch.unsqueeze(timg, 0)

    ################### Random Projections
    RP = RandomProjection(compression_factor, timg.shape)
    y_img = RP(timg)   
    y_img = y_img[0, :].numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(y_img, cmap='gray')
    plt.title("Proxy")
    plt.savefig('rp.jpg')


    ################### Random Sampling Time Domain

    RSTD = RSTD(compression_factor, timg.shape)
    y_img = RSTD(timg)   
    y_img = y_img[0, :].numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(y_img, cmap='gray')
    plt.title("Proxy")
    plt.savefig('rstd.jpg')



    ################### Uniform Sampling Time Domain

    USTD = USTD(compression_factor, timg.shape)
    y_img = USTD(timg)   
    y_img = y_img[0, :].numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(y_img, cmap='gray')
    plt.title("Proxy")
    plt.savefig('ustd.jpg')

    
    #################### Random Sampling Frequency Domain

    RSFD = RSFD(compression_factor, timg.shape)
    y_img = RSFD(timg)   
    y_img = y_img[0, :].numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(y_img, cmap='gray')
    plt.title("Proxy")
    plt.savefig('rsfd.jpg')


    ################### Low-Frequency Sampling

    LFS = LFS(compression_factor, timg.shape)
    y_img = LFS(timg)   
    y_img = y_img[0, :].numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(y_img, cmap='gray')
    plt.title("Proxy")
    plt.savefig('lfs.jpg')



    ################## Equispaced frequency sampling

    EFS = EFS(compression_factor, timg.shape)
    y_img = EFS(timg)   
    y_img = y_img[0, :].numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(y_img, cmap='gray')
    plt.title("Proxy")
    plt.savefig('efs.jpg')