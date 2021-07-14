import cv2
import numpy as np

PATH='testcases\\'
NAME='clr4.png'
'''
PATH='D:\\lab2\\'
NAME='origin1.png'
'''
img=cv2.imread(PATH+NAME,0)
original = img.copy()
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
img = cv2.LUT(img, table)
cv2.imshow("original", original)
cv2.imshow("Output", img)
cv2.imwrite(PATH+NAME+'-stretch.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows() 