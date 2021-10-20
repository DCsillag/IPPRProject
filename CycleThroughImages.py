import cv2 as cv
import os
DIR = r"dataset\\"

# Update two values based on image ranges
for i in range(400,401):
    cv.imshow(str(i)+" IMG", cv.imread(DIR+str(i)+".jpg"))
    cv.waitKey(0)

# Label Categories 
# Good : g
# Average : a
# bad : b
# Terrible : t


