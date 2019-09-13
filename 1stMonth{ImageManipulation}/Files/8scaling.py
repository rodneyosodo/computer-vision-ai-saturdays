import cv2

img = cv2.imread('../Images/jurassicworld.jpg',cv2.IMREAD_COLOR)

res = cv2.resize(img,None,fx=3, fy=3, interpolation = cv2.INTER_LINEAR)
#OR
# height, width = img.shape[:2]
# res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_LINEAR

cv2.imwrite('../Images/8scaling.jpg', res)
cv2.imshow('img',res)
cv2.waitKey(0)
cv2.destroyAllWindows()