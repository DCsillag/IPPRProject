import cv2 as cv #import OpenCV

img = cv.imread('dataset/142.jpg') #reads in image

cv.imshow('142', img) #displays image in new window

cv.waitKey(0) #waits for a key to be pressed
