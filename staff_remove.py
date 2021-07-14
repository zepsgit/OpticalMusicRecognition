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

def rle_remove(bi_img):
    segmenter = Segmenter(bi_img)
    imgs_with_staff = segmenter.regions_with_staff
    imgs_without_staff = segmenter.regions_without_staff
    for i, img in enumerate(imgs_without_staff):
        show_images([img, imgs_with_staff[i]])
        cv2.imwrite('testcases\\('+str(i)+').png', img*255)

