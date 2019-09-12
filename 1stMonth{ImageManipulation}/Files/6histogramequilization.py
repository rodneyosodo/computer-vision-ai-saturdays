import numpy as np
import cv2
import math

def histogram(img):
    height = img.shape[0]
    width = img.shape[1]
    
    hist = np.zeros((256))

    for i in np.arange(height):
        for j in np.arange(width):
            a = img.item(i,j)
            hist[a] += 1
            
    return hist

def cumulative_histogram(hist):
    cum_hist = hist.copy()
    
    for i in np.arange(1, 256):
        cum_hist[i] = cum_hist[i-1] + cum_hist[i]
        
    return cum_hist

img = cv2.imread('../Images/logan.jpg', cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]
pixels = width * height

hist = histogram(img)
cum_hist = cumulative_histogram(hist)

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        b = int(cum_hist[a] * 255.0 / pixels)
        img.itemset((i,j), b)

cv2.imwrite('../Images/6hist.jpg', img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()