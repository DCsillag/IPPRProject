import pandas as pd
import numpy as np
from LoadShowBoundingBox import *
import ImagePipeline_Contour as im
import cv2 as cv

from ImagePipeline_sift import processImg as processImg_sift

num = 394
img = getImage(num)
ref_img = cv.imread('standardPlate.jpeg')
pred_cords = processImg_sift(img, ref_img)

cv.imshow('Sift_Predict', ShowPredBox(img, pred_cords, num))


cv.waitKey(0)

#showImage(60)