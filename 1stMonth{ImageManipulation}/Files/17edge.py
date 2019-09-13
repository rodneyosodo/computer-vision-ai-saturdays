import numpy as np
import cv2

img = cv2.imread('../Images/love.jpg',cv2.IMREAD_COLOR)
edges = cv2.Canny(img,100,200)

cv2.imwrite('../Images/17edge.jpg', edges)
cv2.imshow('image',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()