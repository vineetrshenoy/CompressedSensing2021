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

class RandomRate(object):


    def __init__(self, min_rate, max_rate, dist, img_shp):
        
        self.rng = np.random.default_rng(seed=21) #Set RNG for repeatble results
        self.N =  img_shp[1] *img_shp[2] #length of vectorized image

        A = self.rng.standard_normal((self.N, self.N)) #sensing matrix
        A = np.transpose(scipy.linalg.orth(np.transpose(A)))
        self.A = torch.from_numpy(A).type(torch.FloatTensor) 

        self.dist = dist


        self.min_rate = min_rate
        self.max_rate = max_rate


    def __call__(self, tensor):

        shape = tensor.shape

        if self.dist == 'beta':
            dist_vals = self.rng.beta(2,5, 1)
            dist_vals = dist_vals * (self.max_rate - self.min_rate) + self.min_rate


        elif self.dist == 'exponential':
            beta = -1 * self.max_rate / np.log(0.01)
            dist_vals = self.rng.exponential(beta, 1)
            dist_vals = dist_vals + self.min_rate*np.ones(dist_vals.shape)
            excess = np.where(dist_vals > self.max_rate)
            dist_vals[excess] = self.max_rate

        meas_rate = np.floor(self.N * dist_vals)
    

        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector       


        A = self.A[0:int(meas_rate[0]), :] #Gets the measurement matrix
        y = torch.matmul(A, x.type(torch.FloatTensor)) #get measurements
        Atran = torch.transpose(A, 0, 1)

        proxy = torch.matmul(Atran, y) #proxy image
        proxy = torch.transpose(proxy, 0, 1)
        proxy = torch.reshape(proxy, shape)
        
        return proxy





class VRate(object):


    def __init__(self, min_rate, max_rate, num_rates, dist, img_shp):
        
        self.rng = np.random.default_rng(seed=21) #Set RNG for repeatble results
        self.N =  img_shp[1] *img_shp[2] #length of vectorized image

        A = self.rng.standard_normal((self.N, self.N)) #sensing matrix
        A = np.transpose(scipy.linalg.orth(np.transpose(A)))
        self.A = torch.from_numpy(A).type(torch.FloatTensor) 

        self.dist = dist


        self.min_rate = min_rate
        self.max_rate = max_rate
        self.num_rates = num_rates


    def __call__(self, tensor):

        shape = tensor.shape

        if self.dist == 'beta':
            dist_vals = self.rng.beta(2,5, self.num_rates)
            dist_vals = dist_vals * (self.max_rate - self.min_rate) + self.min_rate


        elif self.dist == 'exponential':
            beta = -1 * self.max_rate / np.log(0.01)
            dist_vals = self.rng.exponential(beta, self.num_rates)
            dist_vals = dist_vals + self.min_rate*np.ones(dist_vals.shape)
            excess = np.where(dist_vals > self.max_rate)
            dist_vals[excess] = self.max_rate

        meas_rate = np.floor(self.N * dist_vals)
        row = meas_rate.shape[0]


        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector       
        #data_mat = []
        for i in range(0, row):

            A = self.A[0:int(meas_rate[i]), :] #Gets the measurement matrix
            y = torch.matmul(A, x.type(torch.FloatTensor)) #get measurements
            Atran = torch.transpose(A, 0, 1)

            proxy = torch.matmul(Atran, y) #proxy image
            proxy = torch.reshape(proxy, shape)
            #proxy = torch.unsqueeze(proxy, 0)
            #data_mat.append(proxy)
            
            if i == 0:
                data_mat = proxy
            else:
                data_mat = torch.cat((data_mat, proxy))
            
            #data_mat = torch.squeeze(data_mat)

        #data_mat = torch.stack(data_mat)
        return data_mat


class RandomRateTesting(CSTransform):


    def __init__(self, compression_factor, img_shp):
        
        super().__init__(compression_factor, img_shp)

        self.rng = np.random.default_rng(seed=21) #Set RNG for repeatble results
        self.N =  img_shp[1] *img_shp[2] #length of vectorized image

        A = self.rng.standard_normal((self.N, self.N)) #sensing matrix
        A = np.transpose(scipy.linalg.orth(np.transpose(A)))
        self.A = torch.from_numpy(A).type(torch.FloatTensor) 
        self.compression_factor = compression_factor


    def __call__(self, tensor):

        shape = tensor.shape
        M = np.floor(self.N * self.compression_factor)
        
        A = self.A[0:int(M), :]

        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector
        y = torch.matmul(A, x.type(torch.FloatTensor)) #get measurements

        Atran = torch.transpose(A, 0, 1)

        proxy = torch.matmul(Atran, y) #proxy image
        proxy = torch.transpose(proxy, 0, 1)
        proxy = torch.reshape(proxy, shape)

        
        return proxy


   
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


class RSTD(CSTransform): #Random Sampling Time Domain

    def __init__(self, compression_factor, img_shp):
 
        super().__init__(compression_factor, img_shp)
        
        
        self.A = np.eye(self.N)
        all_idx = np.arange(0, self.N, 1)
        idx = self.rng.choice(self.N, self.M, replace=False)
        dif = np.setdiff1d(all_idx, idx)
        self.A[dif, :] = 0
        self.A = torch.from_numpy(self.A).type(torch.FloatTensor)

        measmat = np.eye(self.N)
        self.measmat = measmat[idx, :]
        self.measmat = torch.from_numpy(self.measmat).type(torch.FloatTensor)
        '''
        ort = np.transpose(scipy.linalg.orth(np.transpose(self.measmat)))

        #returns false, locations of nonzero are same, but not values
        flag = np.array_equal(self.measmat, ort) 
        
        measmat_nz = np.nonzero(self.measmat)
        ort_nz = np.nonzero(ort)

        print(self.measmat[measmat_nz[0][0], measmat_nz[1][0]])
        print(ort[ort_nz[0][0], ort_nz[1][0]])

        
        '''
    def __call__(self, tensor):

        shape = tensor.shape
        x = torch.flatten(tensor, 1).transpose(0,1) #get image as vector
        y = torch.matmul(self.A, x.type(torch.FloatTensor)) #get measurements
        y_measmat = torch.matmul(self.measmat, x.type(torch.FloatTensor))
        
        img_measmat =  torch.matmul(torch.transpose(self.measmat, 0, 1), y_measmat) 
        img_measmat = torch.reshape(img_measmat, shape)
        #img = torch.reshape(y, shape)

        return img_measmat
    
    
    
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


    ################### Random Rates
    VR = RandomRate(0.01, 0.25, 'beta', timg.shape)
    y_img = VR(timg)   
    y_img = y_img[0, :].numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(y_img, cmap='gray')
    plt.title("Proxy")
    plt.savefig('rp.jpg')


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