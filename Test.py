import pandas as pd
import numpy as np

import cv2 as cv

img = cv.imread("dataset/train/142.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edge = cv.Canny(gray, 5,5)

cv.imshow("142", img)
cv.imshow("142_gray", gray)
cv.imshow("Canny", edge)

cv.waitKey(0)