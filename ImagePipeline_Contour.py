# Import Standard Python Libraries
import cv2 as cv
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import imutils as im

# Create Image processing pipeline function

def processImg(img):
    # Convert Grayscale, bilateral and create canny edge detection. 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bilateral = cv.bilateralFilter(gray, 11, 17, 17)
    edged = cv.Canny(bilateral, 30, 200)

    # Find Contours based on edges
    keypoints = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #chainapproxsimple will get simplified ver of contour (approximate two keypoints)
    contours = im.grab_contours(keypoints) 
    contours = sorted(contours, key = cv.contourArea, reverse = True)[0:10]

    # Loop through each contour to see if there are any square shapes identified. 
    location = None
    
    for contour in contours:
        approx = cv.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    # Mask Image to identify predicted Bounding Box on the image
    mask = np.zeros(gray.shape, np.uint8) 
    try:
        lp = cv.drawContours(mask,[location],0,255,-1,)
        lp = cv.bitwise_and(img,img,mask=mask)
    except:
        return [0,0,0,0]

    # Get Bounding Box from Image
    (x,y) = np.where(mask==255)
    # Note this was reversed to match input data dimensions from xml
    return [np.min(y), np.min(x), np.max(y), np.max(x)]

