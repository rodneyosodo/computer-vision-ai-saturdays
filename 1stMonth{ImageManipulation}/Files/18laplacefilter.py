import numpy as np
import cv2

img = cv2.imread('../Images/logan.jpg', cv2.IMREAD_GRAYSCALE)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1]

laplace = (1.0/16) * np.array(
        [[0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]])
sum(sum(laplace))

for i in np.arange(2, height-2):
    for j in np.arange(2, width-2):        
        sum = 0
        for k in np.arange(-2, 3):
            for l in np.arange(-2, 3):
                a = img.item(i+k, j+l)
                w = laplace[2+k, 2+l]
                sum = sum + (w * a)
        b = sum
        img_out.itemset((i,j), b)

cv2.imwrite('../Images/18laplacefilter.jpg', img_out)

cv2.imshow('image',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()