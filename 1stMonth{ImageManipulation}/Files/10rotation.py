import numpy as np
import cv2

img = cv2.imgread('../Image/restore.jpg', cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]

for i in np.arange(height):
    for j in np.arange(width):
        pass

# ROTATION
# cols-1 and rows-1 are the coordinate limits.
# cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
# cv.warpAffine(img,M,(cols,rows))
