import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#gaussian kernel generator
def gkern(l, sig):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

#kernel generator
'''
type='average'|'gaussian'|'sobel_h'|'sobel_v'|'laplacian'|'median'
size=odd number like 3,5,7...
sigma >=0
'''
def knl_generator(type, size, sigma):
    if type=='average':
        return np.ones([size,size],dtype=int)/(size*size)
    if type=='gaussian':
        return gkern(size, sigma)
    if type=='sobel_v':
        return np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
    if type=='sobel_h':
        return np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    if type=='laplacian':
        return np.array([[0.4038,    0.8021,    0.4038],
                        [0.8021,   -4.8233,    0.8021],
                        [0.4038,    0.8021,    0.4038]])
    if type=='median':
        return np.zeros([size,size])

'''
function: spatial filter including linear and non-linear filter
input:img--np.array|type--'linear' or 'nonlinear'|median_size--the size of median filter
output:filtered image--np.array
'''
def filter(img, type, kx):
    m,n=img.shape
    size=kx.shape[0]
    pad_len=int((size-1)/2)
    new_img=np.zeros([m,n],dtype=np.uint8)
    padded_img=pad_img(img, size)
    for i in range(pad_len,m+pad_len):
        for j in range(pad_len,n+pad_len):
            sum_prod=0
            for s in range(0,size):
               for t in range(0,size):
                   p=i-pad_len+s
                   q=j-pad_len+t
                   if type=='linear':
                       sum_prod=sum_prod+padded_img[p,q]*kx[s,t]
                   if type=='nonlinear':
                       kx[s,t]=padded_img[p, q]
            if type=='linear':
                new_img[i-pad_len,j-pad_len]=sum_prod
            if type=='nonlinear':
                sorted_kx=np.sort(kx,axis=None,kind='quicksort')
                med=sorted_kx[int((size*size-1)/2)]
                new_img[i-pad_len, j-pad_len]=med
    return new_img

#pad image border with 0
#size denote kernel size
def pad_img(img,size):
    m,n=img.shape
    pad_len=int((size-1)/2)#padding length for each edge
    padded_img=np.zeros([m+size-1,n+size-1])
    for i in range(pad_len, pad_len+m):
        for j in range(pad_len, pad_len+n):
            padded_img[i,j]=img[i-pad_len,j-pad_len]
    return padded_img
