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
type='average'|'gaussian'|'sobel_h'|'sobel_v'|'laplacian'
size=odd number like 3,5,7...
sigma >=0
'''
def knl_generator(type, size, sigma):
    if type=='average':
        return np.ones([size,size],dtype=int)/size*size
    if type=='gaussian':
        return gkern(size, sigma)
    if type=='sobel_h':
        return np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
    if type=='sobel_v':
        return np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    if type=='laplacian':
        return np.array([[0.4038,    0.8021,    0.4038],
                        [0.8021,   -4.8233,    0.8021],
                        [0.4038,    0.8021,    0.4038]])

def linear_filter(kx, img):
    m,n=img.shape
    kernel_n=kx.shape[0]
    kn=int((kernel_n-1)/2)
    img_=np.zeros([m,n])
    img_new=np.zeros([m+kernel_n,n+kernel_n])
    for s in range(kn,m+kn):
        for t in range(kn,n+kn):
            img_new[s,t]=img[s-kn,t-kn]
            sum_p=0
            for p in range(0,kernel_n):
               for q in range(0,kernel_n):
                    sum_p=sum_p+img_new[s-kn+p,t-kn+q]*kx[p,q]
                    img_[s-kn,t-kn]=sum_p
    return img_

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

#non-linear filter:median filter
def medf(size,img):
    m,n=img.shape
    kernel=np.zeros([size,size])
    pad_len=int((size-1)/2)
    padded_img=pad_img(img, size)
    new_img=np.zeros([m,n],dtype=np.uint8)
    for i in range(pad_len,pad_len+m):
        for j in range(pad_len,pad_len+n):
            for s in range(0,size):
                for t in range(0,size):
                    p=i-pad_len+s
                    q=j-pad_len+t
                    kernel[s,t]=padded_img[p,q]
                    #print('s',s,'t',t, 'i',p,'j',q)
            sorted_knl=np.sort(kernel, axis=None, kind='quicksort')
            #print(sorted_knl)
            med=sorted_knl[int((size*size-1)/2)]
            new_img[i-pad_len,j-pad_len]=med
    return new_img
    