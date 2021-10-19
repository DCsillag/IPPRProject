import cv2 as cv
import numpy as np

#read in image from dataset
img = cv.imread('dataset/9.jpg')
cv.imshow('9', img)


#convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#Blur and reduce noise- note there are a number
#using Guassian
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT) #the (3,3) is the kernel size and must be odd, to increase the blur increase the kernel size i.e. (7,7)
cv.imshow('Blur', blur)

#create an edge cascade- find edges present
#canny edge detector 
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)


#dilate an image using a specific structuring element (the canny edges)
dilated = cv.dilate(canny, (3,3), iterations=1) #there can be several iterations, and don't forget (3,3) is the kernel size
cv.imshow('Dilated', dilated)

#can erode the image to get back the structuring element 
#eroding 
eroded = cv.erode(dilated, (3,3), iterations=1) #match the dilated edges and get back to the same area from dilation
cv.imshow('Eroded', eroded)

cv.waitKey(0)