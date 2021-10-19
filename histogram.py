import cv2 as cv
import matplotlib.pyplot as plt
from numpy import histogram

#histogram computation- visualise the distribution of pixel intensities in an image

img = cv.imread('dataset/9.jpg')
cv.imshow('9', img)

#can compute histogram for both RGB and grayscale images

#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

#grayscale histogram
#need to pass in a list of images- not just one, therefore the []
#gray_hist = cv.calcHist([gray], [0], None, [256], [0,256])

#plt.figure()
#plt.title('Grayscale Histogram')
#plt.xlabel('Bins')
#plt.ylabel('# of pixels')
#plt.plot(gray_hist)
#plt.xlim([0,256])
#plt.show()


#colour histogram
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

#plt.show()




cv.waitKey(0)