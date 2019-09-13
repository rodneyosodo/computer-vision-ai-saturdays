import cv2
import numpy as np
 
img = cv2.imread('../Images/logan.jpg', cv2.IMREAD_GRAYSCALE)
img = img/255.0
img_power_law_transformation = cv2.pow(img,1.2)

cv2.imwrite('../Images/12powerlaw.jpg', img_power_law_transformation)
cv2.imshow('Original Image',img)
cv2.imshow('Power Law Transformation',img_power_law_transformation)
cv2.waitKey(0)