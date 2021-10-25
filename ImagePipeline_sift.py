import cv2 as cv
import os
import numpy as np

# Fetch Z_score of a keypoint list.

def z_score(coords_list):
    #using the z-score algoritm to determine outliners in keypoints identified above
    mean_y = np.mean(coords_list)
    stddev_y = np.std(coords_list)
    z_scores = np.abs([(y - mean_y)/stddev_y for y in coords_list])
    return z_scores

# Use Emily's Sift.py to build a function for use in existing image pipelines

def processImg(img, reference_img):
    # Convert Image to GrayScale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Create Sift detector
    sift = cv.xfeatures2d.SIFT_create()

    # Get KeyPoints
    kp1, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(reference_img,None)

    #define parameters for flann matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
    search_params = dict(checks = 100)

    #flann matcher object creation
    flann = cv.FlannBasedMatcher(index_params, search_params)

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
    
    #determining the outliners to remove from keypoints based on threshold
    thresh = .975
    outliers = [(t[0]>thresh or t[1]>thresh) for t in z_score(keypoints)]
    cleaned_kps = []
    for i, kp in enumerate(keypoints):
        if not outliers[i]:
            cleaned_kps.append(kp)

    #looping through cleaned keypoints to identify region of interest based on min and max x and ys
    try:
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
    except:
        min_x, min_y, max_x, max_y = 0, 0, 0, 0

    return [min_x, min_y, max_x, max_y]

