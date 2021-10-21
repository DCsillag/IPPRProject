import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#file path
lp_Cascade = cv.CascadeClassifier('./datset.xml')

def lp_detect(img, text=''):  #detects & performs blurring of number plate
    lp_img = img.copy()
roi = img.copy()
# detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
lp_Rect = lp_Cascade.detectMultiScale(lp_Img, scaleFactor = 1.2, minNeighbors = 7) 

for (x,y,w,h) in lp_Rect:
    roi_ = roi[y:y+h, x:x+w, :]  #extracting the Region of Interest of lp for blurring.
    lp_plate = roi[y:y+h, x:x+w, :]
    cv.rectangle(lp_img,(x+2,y),(x+w-3, y+h-5),(0,255,0),3)
if text!='':
     lp_img = cv.putText(lp_img, text, (x-w//2,y-h//2), 
     cv.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (51,181,155), 1, cv.LINE_AA)
return lp_img, lp_plate

def display_img(img):
    img_ = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    plt.imshow(img_)
    plt.show()
    
inputimg = cv.imread('./dataset/50.jpg')
inpimg, lp = lp_detect(inputimg)
display_img(inpimg)

display_img(lp)
#read in image from dataset
img = cv.imread('./dataset/50.jpg')
cv.imshow('50', img)

blank = np.zeros(img.shape, dtype='uint8')
#cv.imshow('Blank', blank)

#convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
bilateral = cv.bilateralFilter(gray, 13, 15, 15)
cv.imshow('Bilateral', bilateral)

#create an edge cascade- find edges present
#canny edge detector 
edged = cv.Canny(bilateral, 170, 200)
cv.imshow('Canny Edges', edged)

#find contours based on edges
contours = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
contours = sorted(contours, key = cv.contourArea, reverse = True)[:90] #all contoiurs less than  are discarded
NumerPlateCnt = None #donthave numberplate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            x,y,w,h = cv.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            break

if NumberPlateCnt is not None:
    # Drawing the selected contour on the original image
    cv.drawContours(img, [NumberPlateCnt], -1, (0,255,0), 3)





cv.waitKey(0)