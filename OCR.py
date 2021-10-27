import pytesseract as pt
import cv2 as cv
import numpy as np
import re

# Define Path for Tesseract.exe, will depend on local user
pt.pytesseract.tesseract_cmd = r'C:\Users\danie\tesseract.exe'
custom_config = r'--oem 3 --psm 6'

# Given a cropped image of the license plate, return tesseract image-to-string detection prediction
def getPlateChars(cropped_img, thresh=128):
    ret, bin_img = cv.threshold(cropped_img, thresh, 255, cv.THRESH_BINARY)
    text = pt.image_to_string(bin_img, config=custom_config, lang='eng')
    text = re.sub('[^0-9] ', '', text)
    text = text.split()
    text = [re.sub('[^0-9]', '', w) for w in text]
    try:
        text = str(text[0]) + str(text[-1])
    except:
        text = str(text)
    return text


def cropImage(img, cords):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img[cords[1]:cords[3]+1, cords[0]:cords[2]+1]