import numpy as np
import cv2
import math

img = cv2.imread('../Images/bottle.jpg', cv2.IMREAD_ANYCOLOR)
height = img.shape[0]
width = img.shape[1]

contrast = 5

for i in np.arange(height):
    for j in np.arange(width):
        for k in range(0,3,1):
            a = img.item(i,j,k)
            b = math.ceil(a * contrast)
            if b > 255:
                b = 255
            img.itemset((i,j,k), b)

cv2.imwrite('../Images/1contrast.jpg', img)

cv2.imshow('image2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()