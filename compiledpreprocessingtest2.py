import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils as im

#read in image from dataset and convert to grayscale
img = cv.imread('./dataset/50.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(cv.cvtColor(gray,cv.COLOR_BGR2RGB))

# Noise removal with bilateral filter(removes noise while preserving edges) + find edges present
bilateral = cv.bilateralFilter(gray, 11, 17, 17)
edged = cv.Canny(bilateral, 30, 200)
plt.imshow(cv.cvtColor(edged,cv.COLOR_BGR2RGB))

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
plt.imshow(cv.cvtColor(lp, cv.COLOR_BGR2RGB))

#crop segment
(x,y) = np.where(mask==255) #find section of image that isnt blacked out, get set of coordinates that arent masked over
(x1, y1) = (np.min(x), np.min(y)) #onecorner
(X2, y2) = (np.max(x), np.max(y)) #opposing diagonal corner
cropped_lp = gray[x1:x2+1, y1:y2+1] #added 1 to give us a little buffer

plt.imshow(cv.cvtColor(cropped_lp, cv.COLOR_BGR2RGB))




