import cv2 as cv

#read in image from dataset
img = cv.imread('dataset/9.jpg')
cv.imshow('9', img)

#averaging (method of blurring)
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)

#Guassian Blur- similar to averaging (instead of computing the average, each pixel is given a weight and the average of the products of the weight give the new value)- less bluring as opposed to averaging (more natural though)
#standard deviation in the x direction is the last component in the bracket
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Guass Blur', gauss)

#median blur- same as averaging, but instead of average, it finds the median of the surrounding pixels (more effective in reducing noise)
#instead of kernel size an integer is used, computer assumes 3x3 basically
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

#bileratial blur- most effective because of how it blurs, applies bluring but retains the edges in the image
#research into this move- doesn't take kernel but diameter, signma colour, sigma space
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bileratal', bilateral)

cv.waitKey(0)