import os
import numpy as np
import cv2 as cv
from LoadShowBoundingBox import *

# Setting out a general structure to this document

# Define a function that when given an image number, and a predicted coordinate location, return a classification
def imgAccuracy(img_num, pred_cords):
    true_box = getPlateCords(img_num)
    return calculateClassfication(true_box, pred_cords)

# Define a function that evalutes the category of outcome.
# If IoU > Threshold : True Positive, 1
# If IoU < Threshold : False Positive 0
# If IoU 0 Then False Negative 0
def calculateClassfication(iou, thresh):
    return 1 if iou > thresh else 0

# Define a function that when given two sets of coordinates, returns an IoU score.
def calculateIoU(boxA, boxB):
    # Find the dimensions of the internal rectangle.
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of the central rectangle
    innerArea = max(0, xB - xA + 1) * max(0, yB - yA +1)

    # Compute area of each box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute and return intersection of union
    return innerArea / float(boxAArea + boxBArea - innerArea)

# Test
print(imgAccuracy(0, [1200, 2000, 2000, 2300]))


