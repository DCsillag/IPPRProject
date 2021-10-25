import cv2 as cv
import numpy as np
import pandas as pd
import LoadShowBoundingBox as LB

# Use this file to write in the number plate labels for good images
# Write plate details here: 
# https://docs.google.com/spreadsheets/d/1SwTrOaLj07ywT8s3cRcgXeF4jyV4ajHFxo9JfRYogvc/edit#gid=1337144092
# Daniel 0-150
# Emily Reed 150-300
# Felly Li 300-450
# Jordan Roberts 450-600
# Lisa Chan 600-708


df = pd.read_csv("Img_categories.csv")

# Update the range field with the numbrs allocated to you.
# Ensure you write in the field matching the image number, image number 
# is on window and printed to terminal
for i in range(0, 150):
    if df.Category.iloc[i] == 'g':
        LB.showImage(i)
