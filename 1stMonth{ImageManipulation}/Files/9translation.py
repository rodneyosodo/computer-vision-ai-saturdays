import numpy as np
import cv2 as cv2

img = cv2.imread('../Images/jurassicworld.jpg',cv2.IMREAD_COLOR)
height, width = img.shape[:2]

M = np.float32([[1,0,20],[0,1,30]])

img = cv2.warpAffine(img,M,(width,height))

cv2.imwrite('../Images/9Translating.jpg', img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

