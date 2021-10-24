import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import random
import csv

# DIR = r"dataset\\"
DIR = './dataset/'

def sift_detector(new_image, image_template):
    #used to compare input image to template
    #outputs the number of sift matches between two images

    img1 = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
    img2 = image_template

    #create SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    #get keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # print(des1)

    #define parameters for flann matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
    search_params = dict(checks = 100)

    #flann matcher object creation
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #get matches using k-nearest neighbour
    #"matches" represents the number of similar matches found in both images
    matches = flann.knnMatch(des1, des2, k=2)

    #using lowe's ratio test to store good matches
    good_matches = []
    keypoints = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            pt1 = kp1[m.queryIdx].pt
            keypoints.append(pt1)
    return [len(good_matches), keypoints]

def z_score(coords_list):
    #using the z-score algoritm to determine outliners in keypoints identified above
    mean_y = np.mean(coords_list)
    stddev_y = np.std(coords_list)
    z_scores = np.abs([(y - mean_y)/stddev_y for y in coords_list])
    return z_scores

#load reference images template
image_template = cv2.imread('standardPlate.jpeg',0)
    
#create array of just jpg images to loop through
jpg_arr = []    
for file in os.listdir(DIR):
    if file.endswith('.jpg'):
        jpg_arr.append(file)

#read csv with categories of good images
EXCLUDE_CATS = ['a', 'b', 't']
IOU_THRESHOLD = 0.6
img_cats_arr = []
with open('Img_categories.csv', newline='') as img_cats:
    spamreader = csv.reader(img_cats, delimiter=',', quotechar='|')
    for row in spamreader:
        img_cats_arr.append(row)

#create new array of just good images
img_cats_arr_g = []
for elm in img_cats_arr:
    if elm[1] not in EXCLUDE_CATS:
        img_cats_arr_g.append(elm[0])

# for file in jpg_arr[:10]:
for file in random.sample(img_cats_arr_g, 25):
    print('Filename: '+file+'.jpg')
    cur_image = cv2.imread(DIR+file+'.jpg')

    #get height + width of each image
    h, w = cur_image.shape[:2]

    #get sift matches
    sift_res = sift_detector(cur_image, image_template)
    matches = sift_res[0]
    keypoints_found = sift_res[1]

    #determining the outliners to remove from keypoints based on threshold
    thresh = .975
    outliers = [(t[0]>thresh or t[1]>thresh) for t in z_score(keypoints_found)]
    cleaned_kps = []
    for i, kp in enumerate(keypoints_found):
        if not outliers[i]:
            cleaned_kps.append(kp)

    #looping through cleaned keypoints to identify region of interest based on min and max x and ys
    min_x = int(cleaned_kps[0][0])
    min_y = int(cleaned_kps[0][1])
    max_x = 0
    max_y = 0
    for kp in cleaned_kps:
        if kp[0] < min_x:
            min_x = int(kp[0])
        if kp[1] < min_y:
            min_y = int(kp[1])
        if kp[0] > max_x:
            max_x = int(kp[0])
        if kp[1] > max_y:
            max_y = int(kp[1])

    print("Shape:"+str((w,h)))
    print("Bounding box: "+str((min_x,min_y))+", "+str((max_x,max_y)))

    #define threshold to indicate object has been detected
    threshold = 8

    #if matches is > our threshold then object is deteced
    if matches > threshold:
        # cv2.rectangle(cur_image, (top_l_x,top_l_y), (bot_r_x,bot_r_y), (0,255,255),3)
        for x, y in keypoints_found:
            cv2.circle(cur_image, (int(x),int(y)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.rectangle(cur_image, (min_x,min_y), (max_x,max_y), (0,255,255),3)
        cv2.putText(cur_image,'Licence Plate Detected',(50,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    cv2.imshow('Licence Plate Detector using SIFT', cur_image)
    if cv2.waitKey(0) == 13:
        continue