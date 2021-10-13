import os
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as et

DIR = r"dataset\\"

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

# ShowPlateBox returns an image with the bounding box overlayed.
def showPlateBox(filenum):
    img = getImage(filenum)
    bounds = getPlateCords(filenum)
    cv.rectangle(img, (bounds[0],bounds[1]), (bounds[2], bounds[3]), (0,255,0), 3)
    return img

cv.imshow("Car", showPlateBox(201))
cv.waitKey(0)