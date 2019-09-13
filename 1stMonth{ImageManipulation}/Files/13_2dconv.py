import numpy as np
import cv2

img = cv2.imread('../Images/logan.jpg',cv2.IMREAD_COLOR)
kernel = np.ones((4,4),np.float32)/16
img = cv2.filter2D(img,-1,kernel)

cv2.imwrite('../Images/12conv.jpg', img)
cv2.imshow('Image',img)
cv2.waitKey(0)