import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from spatial_filter import knl_generator, filter
from global_thresholding import find_th, bi_image
from skimage.filters import threshold_otsu, threshold_local
from skimage.color import rgb2gray
from skimage.morphology import square, erosion, dilation, closing, opening
from segmenter import Segmenter
from commonfunctions import *
from staff_remove import *

PATH='testcases\\'
NAME='y.png' 

img=cv2.imread(PATH+NAME,0)
img1 = img.copy()#original 
k=knl_generator('median', 3, 0)
img2=filter(img1, 'nonlinear', k)#filtered image
cv2.imwrite(PATH+'img2.png',img2)
myth=find_th(img)
img3=bi_image(img2, myth)#binarize img2
cv2.imwrite(PATH+'img3.png', img3*255)
img4=rle_remove(img3)

