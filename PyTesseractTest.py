import pytesseract as pt
import cv2 as cv
import numpy as np
from LoadShowBoundingBox import *
import matplotlib.pyplot as plt
import imutils as im
import re

pt.pytesseract.tesseract_cmd = r'C:\Users\danie\tesseract.exe'
custom_config = r'--oem 3 --psm 6'

#read in image from dataset and convert to grayscale
img = getImage(89)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Noise removal with bilateral filter(removes noise while preserving edges) + find edges present
bilateral = cv.bilateralFilter(gray, 11, 17, 17)
edged = cv.Canny(bilateral, 30, 200)

#find contours based on edges
keypoints = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #chainapproxsimple will get simplified ver of contour (approximate two keypoints)
contours = im.grab_contours(keypoints) 
contours = sorted(contours, key = cv.contourArea, reverse = True)[0:10] #all contours less 10 than  are discarded

#loop through each contour and actually represent square or number plate
location = None
for contour in contours:
    approx = cv.approxPolyDP(contour, 10, True) #approximate the polygon from contour, 10 specify how accurate or finegrain our approximate is
    if len(approx) == 4: #selects contour w/ 4 corners/keypoints --> most likely our numberplate location 
        location = approx
        break

#masking
mask = np.zeros(gray.shape, np.uint8)  #created blank mask (same shape as og gray image, pass shape of og image, fill in with blank 0s )
lp = cv.drawContours(mask,[location],0,255,-1,) #draw contours in the image, want to draw location
lp = cv.bitwise_and(img,img,mask=mask) #overlap mask, return segment of our iamge ie numberplate

#crop segment
(x,y) = np.where(mask==255) #find section of image that isnt blacked out, get set of coordinates that arent masked over
(x1, y1) = (np.min(x), np.min(y)) #onecorner
(x2, y2) = (np.max(x), np.max(y)) #opposing diagonal corner
cropped_lp = gray[x1:x2+1, y1:y2+1] #added 1 to give us a little buffer

ret, cropped_lp_binary, = cv.threshold(cropped_lp, 128, 255, cv.THRESH_BINARY)

cv.imshow("Image", cropped_lp_binary)
text = pt.image_to_string(cropped_lp_binary, config=custom_config, lang='eng+ara')

text = re.sub('[^0-9] ', '', text)
text = text.split()
text = [re.sub('[^0-9]', '', w) for w in text]
text = str(text[0]) + str(text[-1])
print(text)
cv.waitKey(0)



# Old Code
# image = getImage(394)

# text = pt.image_to_string(image)

# print(text)

# cv.imshow("Image", image)
