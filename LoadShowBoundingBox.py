import os
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as et
import pandas as pd
import random

DIR = r"dataset\\"
#DIR = './dataset/'

# Get Plate Cords Returns the coordinates of a number plate for the i-th image in the folder. 
def getPlateCords(filenum):
    tree = et.parse(DIR+str(filenum)+".xml")
    root = tree.getroot()
    cords = []
    for i in range(4):
        cords.append(int(root[6][4][i].text))
    return cords

# Define a function to load an image based on its number
def getImage(filenum):
    return cv.imread(DIR+str(filenum)+".jpg")

# OverlayPlateBox returns an image with the bounding box overlayed.
def OverlayPlateBox(filenum):
    img = getImage(filenum)
    bounds = getPlateCords(filenum)
    cv.rectangle(img, (bounds[0],bounds[1]), (bounds[2], bounds[3]), (0,255,0), 3)
    return img

def OverlayPlateBox_img(img, number):
    bounds = getPlateCords(number)
    cv.rectangle(img, (bounds[0],bounds[1]), (bounds[2], bounds[3]), (0,255,0), 3)
    return img

# Show plate box with an additional bounding box
def ShowPredBox(img, bounds, num):
    img = OverlayPlateBox_img(img, num)
    cv.rectangle(img, (bounds[0],bounds[1]), (bounds[2], bounds[3]), (0,0,255), 3)
    return img

# Show the image using OpenCV
def showImage(filenum):
    cv.imshow("Car"+str(filenum), OverlayPlateBox(filenum))
    cv.waitKey(0)


# Return a random image based on its type (provide function with 'g', 'a', 'p' or 't')
def loadRandomImage(type):
    imageSelection = pd.read_csv("Img_categories.csv")
    while True:
        img_num = random.randint(0, 708)
        if imageSelection.Category.iloc[img_num] == type:
            print("Image number selected " + str(img_num))
            break
    return getImage(img_num)