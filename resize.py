import cv2 as cv
from LoadShowBoundingBox import getImage
from LoadShowBoundingBox import*

#DIR = r"\\dataset\\"
#for i in range(0,100):
#    img = cv.imread(str(i)+" IMG", cv.imread(DIR+str(i)+".jpg"))
#    print('Original Dimensions : ', img.shape)

img = cv.imread('dataset/99.jpg')
print('Original Dimensions : ', img.shape)

scale_percent = 75 #% of the original image size
width = int(img.shape[1] * scale_percent / 100) #calculates the original width
height = int(img.shape[0] * scale_percent / 100) #calculates the original height
dim = (width, height)

#Use resize method to change the dimensions of the image
resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
print('Resized Dimensions : ', resized.shape)

#Image crop function
img_cropped = img[50:width, 160:height]
print('Cropped Dimensions : ', img_cropped.shape)

#Testt
cv.imshow("Cropped Image", img_cropped)
cv.imshow("Resized Image", resized)
cv.waitKey(0)