import numpy as np
import cv2
threshold = 150

img = cv2.imread('../Images/logan.jpg', cv2.IMREAD_GRAYSCALE)
height = img.shape[0]
width = img.shape[1]

for i in np.arange(height):
    for j in np.arange(width):
            a = img.item(i, j)
            
            if a < threshold:
                a = 0
            img.itemset((i, j), a)

cv2.imwrite('../Images/logan.jpg', img)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()