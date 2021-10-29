import cv2 as cv
import numpy as np

img = cv.imread('dataset/14.jpg') #reads in image
#cv.imshow('14', img) #displays image in new window

#
blank = np.zeros(img.shape, dtype='uint8')
#cv.imshow('Blank', blank)

#convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

#Blur and reduce noise- note there are a number
#using Guassian
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT) #the (3,3) is the kernel size and must be odd, to increase the blur increase the kernel size i.e. (7,7)
cv.imshow('Blur', blur)

#canny edge detection
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

#threshold - looks at an image and attempts to binarise the image
ret, thresh = cv.threshold(canny, 125, 255, cv.THRESH_BINARY) #if the value is below 125 set to 0 (black) and above 255 set to 1 (white), then it's a type so we are using binary
cv.imshow('Thresh', thresh)

#using fine contours method- returns contours and hierachies 
#RETR_TREE if you want all hierarchical contours, RETR_EXTERNAL for external contours, RETR_LIST if you want all the contours in the image
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#cv.findContours - looks at the structuring elements (the edges in the image) and returns two values- contours and hierachires
#contours is a python list of all the coordinates of the contours found in the image
#hierarchies refers to the hierarchical representation of contours, e.g. rectangle, inside a square, inside a circle, 
#cv.RETR_LIST- is a mod in which this fine contours method returns and finds the contours. List returns all the contours in the image, exteral retrieves external contours, tree returns hierachical contours
#cv.CHAIN_APPROX_NONE- how we want to approximate the contour. None just returns all the contours, some poeple prefer to use CHAIN_APPROX_SIMPLE which compresses all the contours returned that makes more sense
print(f'len{(contours)} contour(s) found!') #shows number of contours found

#highlight the edges on the image and draw it on the blank doc in red
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)


cv.waitKey(0) #waits for a key to be pressed