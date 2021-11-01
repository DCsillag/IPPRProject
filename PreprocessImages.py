import cv2 as cv
import numpy as np
from LoadShowBoundingBox import *

# Using Lisa's Method to Sharpen the image
def sharpenImage(img):
    laplician_var = cv.Laplacian(img, cv.CV_64F).var()

    while laplician_var < 25:
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv.filter2D(img, -1, sharpen_kernel)
        laplician_var = cv.Laplacian(img, cv.CV_64F).var()
    
    return img

# add a function for resize
