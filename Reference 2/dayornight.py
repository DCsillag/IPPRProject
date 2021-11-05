import os
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('dataset/392.jpg') #reads in image

def isDay(image, dim=10, thresh=0.5):
    # Resize image to 10x10
    img = cv.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh

# find if image is bright or dark
text = "bright" if isDay(img) else "dark"

# write text on image
cv.putText(img, "{}".format(text), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# show image
plt.figure(figsize=(10,10))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()
