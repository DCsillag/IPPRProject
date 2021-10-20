import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./dataset/50.jpg',cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('./dataset/51.jpg',cv2.COLOR_BGR2GRAY)

#create SIFT detector
orb = cv2.ORB_create() 

#get keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

#create matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#match descriptors from images
matches = bf.match(des1,des2)

#sort descriptors by distance
matches = sorted(matches, key=lambda x:x.distance)

#draw first 10 matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

plt.imshow(img3),plt.show()