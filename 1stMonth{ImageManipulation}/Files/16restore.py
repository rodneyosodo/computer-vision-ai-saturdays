import numpy as np
import cv2

img = cv2.imread("../Images/restore.jpg", cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread('mask2.png',0)
# dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
height = img.shape[0]
width = img.shape[1]

threshold = 150

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        if a > threshold:
            b = 255
        else:
            b = 0
        img.itemset((i,j), b)

cv2.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
