import numpy as np
import cv2

img1 = cv2.imread('../Images/jurassicworld.jpg',cv2.IMREAD_COLOR)
img2 = cv2.imread('../Images/love.jpg',cv2.IMREAD_COLOR)

height = img1.shape[0]
width = img1.shape[1]

alpha = 0.6789

for i in np.arange(height):
    for j in np.arange(width):
        for k in range(0,3,1):
            a1 = img1.item(i,j,k)
            a2 = img2.item(i,j,k)
            b = alpha * a1 + (1-alpha) * a2
            img1.itemset((i,j,k), b)

cv2.imwrite('../Images/7linear_blend.jpg', img1)
# cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()