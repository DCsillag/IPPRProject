# Standard Python Library Imports
import numpy as np
import cv2 as cv
import pandas as pd
import os
import sklearn as sk
from sklearn.metrics import accuracy_score

# Custom Code Imports 
from Evaluate import *
from LoadShowBoundingBox import *
from ImagePipeline_Contour import processImg as processImg_contour
from ImagePipeline_sift import processImg as processImg_sift
from OCR import *

# Reference plate, for Sift case
ref_img = cv.imread('standardPlate.jpeg')

EXCLUDE_CATS = ['a', 'b', 't']
IOU_THRESHOLD = 0.6
Image_Info = pd.read_csv('Img_categories.csv')

# Filtered by excluded categories, create the sub_sample of the dataset.
ImageDataset = []
for i in range(Image_Info.shape[0]):
    if Image_Info.Category.iloc[i] not in EXCLUDE_CATS:
        ImageDataset.append([str(i),i,0,0,0,0,1,0,0,0])


# Retrieve the true and predicted bounding box for all images that match the classification condition.
# The ImgData[3] term defines which modelling technique is used for license plate detection, change between sift, contour
# If the plate is detected, pass image through to PyTesseract 
for i, imgdata in enumerate(ImageDataset):
    img = getImage(imgdata[1])
    imgdata[2] = getPlateCords(imgdata[1])
    imgdata[3] = processImg_sift(img, ref_img)
    imgdata[4] = calculateIoU(imgdata[2], imgdata[3])
    imgdata[5] = calculateClassfication(imgdata[4], IOU_THRESHOLD)
    if imgdata[5] == 1:
        croppedImage = cropImage(img, imgdata[3])
        imgdata[7] = getPlateChars(croppedImage)
        if imgdata[7] == imgdata[8]:
            imgdata[9] = 1
    

# Present data in a Dataframe format to increase its readability in the terminal
imageData_df = pd.DataFrame(ImageDataset, 
    columns=['StringNumber','Number', 'trueBox','predBox','IOU','Detected','TrueClass','Pred_text','True_text', 'txt_corr']
)
print(imageData_df.head())
print(imageData_df.Detected.value_counts())

print(imageData_df[(imageData_df['Pred_text'] != 0)].head(10))

# Debug Code Only
# num = 9
# img = getImage(num)
# predicted_cords = processImg(img)
# img = OverlayPlateBox_img(img, num)
# img = ShowPredBox(img, predicted_cords, num)

# cv.imshow("Image both boxes", img)

# cv.waitKey(0)