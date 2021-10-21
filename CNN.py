import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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