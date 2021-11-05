import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv #import OpenCV

from skimage.exposure import is_low_contrast

img = cv.imread('dataset/255.jpg') #reads in image

def isContrast(img, fraction_threshold=0.40):
    # Convert color space to LAB format and extract L channel
    L, A, B = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > fraction_threshold

# find if image is bright or dark
text = "High Contrast" if isContrast(img) else "Low Contrast"

# write text on image
cv.putText(img, "{}".format(text), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# show image
plt.figure(figsize=(10,10))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()



# Check if it is low contrast or not
#out = is_low_contrast(img, fraction_threshold=0.3)
 
# if true print low contrast otherwise high contrast
#if out:
 #   print('image has low contrast')
#else:
 #   print('image has high contrast')

#cv.imshow('142', img) #displays image in new window

