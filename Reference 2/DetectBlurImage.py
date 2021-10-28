import cv2 as cv
import numpy as np

img = cv.imread('dataset/285.jpg', cv.IMREAD_GRAYSCALE)


laplician_var = cv.Laplacian(img, cv.CV_64F).var() #laplician is a filter for kernal convolution method


if laplician_var < 25:
    # Do whatever to adjust it.
    print("Low Quality Image")


print(laplician_var)  #low vlaue = low sharpness(blur)


cv.imshow('285', img)

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv.filter2D(img, -1, sharpen_kernel)

cv.imshow('sharpen', sharpen)

#cv.imshow('laplician', laplician)
cv.waitKey(0)
cv.destroyAllWindows()


#REF https://www.youtube.com/watch?v=5YP7OoMhXbM&ab_channel=Pysource
#REF https://stackoverflow.com/questions/58231849/how-to-remove-blurriness-from-an-image-using-opencv-python-c