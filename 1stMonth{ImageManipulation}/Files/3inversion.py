import numpy as np
import cv2

img = cv2.imread('../Images/bottle.jpg', cv2.IMREAD_ANYCOLOR)

height = img.shape[0]
width = img.shape[1]

max_intensity = 255

for i in np.arange(height):
    for j in np.arange(width):
        for k in range(0,3,1):
            a = img.item(i,j,k)
            b = max_intensity - a
            img.itemset((i,j,k), b)

cv2.imwrite('../Images/3inversion.jpg', img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()