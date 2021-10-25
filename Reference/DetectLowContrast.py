import cv2 as cv
import numpy as np
import argparse as ag
import imutils as im
from skimage.exposure import is_low_contrast 
#used to examine image's histogram and then determining if the range of brightness spans less than a fractional amount


#construct argument parser
ap = ag.ArgumentParser()
ap.add_argument("-t", "--thresh", type=float, default = 0.30, help = "threshold for low contrast")
#image to be considred low cast if it is less than 30% of the range of brightness
args = vars(ap.parse_args())

#image paths, and convert to gray
img = cv.imread('dataset/50.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#blur and edge image
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(blurred, 30, 200)

#intialise the text and color of input image, assuming that image is not low contrast
text = "Not low contrast image"
color = (0, 255, 0)

#detects if low contrast
if is_low_contrast (gray, fraction_threshold=args["thresh"]):
    #update the text and color if so
    text = "Low contrast image"
    color = (0, 0, 255)
    
else:
    keypoints = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #chainapproxsimple will get simplified ver of contour (approximate two keypoints)
    contours = im.grab_contours(keypoints) 
    contours = sorted(contours, key = cv.contourArea, reverse = True)[0:10] #all contours less 10 than  are discarded

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

#draw text on output image
cv.putText(lp, text, (5,25), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, color, 2)

#show image
cv.imshow()

cv.waitKey(0)

#update: threshold is not working, nothing working unless you guys can figure out whats wrong. Reference link is below 
#reference: https://www.pyimagesearch.com/2021/01/25/detecting-low-contrast-images-with-opencv-scikit-image-and-python/ 
