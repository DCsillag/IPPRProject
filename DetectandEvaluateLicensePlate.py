# Standard Python Library Imports
import numpy as np
import cv2 as cv
import pandas as pd
import os
import sklearn as sk

# Custom Code Imports 
from Evaluate import *
from LoadShowBoundingBox import *
from ImagePipeline1 import *

EXCLUDE_CATS = ['a', 'b', 't']
IOU_THRESHOLD = 0.6
Image_Info = pd.read_csv('Img_categories.csv')

# Filtered by excluded categories, create the sub_sample of the dataset.
ImageDataset = []
for i in range(Image_Info.shape[0]):
    if Image_Info.Category.iloc[i] not in EXCLUDE_CATS:
        ImageDataset.append([str(i),i,0,0,0,0])


# Retrieve the true and predicted bounding box for all images that match the classification condition.
for i, imgdata in enumerate(ImageDataset):
    img = getImage(imgdata[1])
    imgdata[2] = getPlateCords(imgdata[1])
    imgdata[3] = processImg(img)
    imgdata[4] = calculateIoU(imgdata[2], imgdata[3])
    imgdata[5] = calculateClassfication(imgdata[4], IOU_THRESHOLD)


imageData_df = pd.DataFrame(ImageDataset, columns=['StringNumber','Number', 'trueBox','predBox','IOU','Classification'])
print(imageData_df.head())

print(imageData_df[(imageData_df.Number == 394)])
print(imageData_df.Classification.value_counts())


# num = 9
# img = getImage(num)
# predicted_cords = processImg(img)
# img = OverlayPlateBox_img(img, num)
# img = ShowPredBox(img, predicted_cords, num)

# cv.imshow("Image both boxes", img)

# cv.waitKey(0)