import cv2 as cv
from LoadShowBoundingBox import getImage
from LoadShowBoundingBox import*

#DIR = r"\\dataset\\"
#for i in range(0,100):
#    img = cv.imread(str(i)+" IMG", cv.imread(DIR+str(i)+".jpg"))
#    print('Original Dimensions : ', img.shape)

img = cv.imread('dataset/100.jpg')
print('Original Dimensions : ', img.shape)

scale_percent = 75 #% of the original image size
width = int(img.shape[1] * scale_percent / 100) #calculates the original width
height = int(img.shape[0] * scale_percent / 100) #calculates the original height
dim = (width, height)

#Use resize method to change the dimensions of the image
resized = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
print('Resized Dimensions : ', resized.shape)

#Image crop function
img_cropped = img[10:width, 10:height]
print('Cropped Dimensions : ', img_cropped.shape)
img_cropped2 = img[20:width, 20:height]
img_cropped3 = img[100:1200, 0:1500]
img_cropped4 = img[200:700, 120:600] #Optimal crop dimensions
#Testt
cv.imshow("Original Image", img)
cv.imshow("Cropped Image", img_cropped)
cv.imshow("Cropped Image 2", img_cropped2)
cv.imshow("Cropped Image 3", img_cropped3)
cv.imshow("Cropped Image 4", img_cropped4)
cv.imshow("Resized Image", resized)
cv.waitKey(0)