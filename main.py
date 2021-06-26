import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from spatial_filter import knl_generator, filter

#test linear filter on 'salt-pepper noise' corrupted image

k=knl_generator('laplacian', 0, 0)
img=cv2.imread('D:\\lab2\\img2.png',0)
cv2.imshow('02', img)
img=filter(img,'linear',k)
cv2.imshow('02-3f',img)
#cv2.imwrite('02-3f_average_3.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#test median filter on 'salt-pepper noise'
img=cv2.imread('testcases\\02.png',0)
img=medf(3,img)
cv2.imshow('02-f', img)
cv2.imwrite('testcases\02-f_median_3.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''