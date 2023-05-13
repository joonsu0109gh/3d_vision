import cv2
import numpy as np
from matplotlib import pyplot as plt
from matchPics import matchPics
from helper import plotMatches
from planarH import computeH_ransac, compositeH

#Q10
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

# Load images
# Detect keypoints and extract descriptors
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(cv_cover, None)
kp2, des2 = orb.detectAndCompute(cv_desk, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches_cv = bf.match(des1, des2)

# Sort matches by distance
matches_cv = sorted(matches_cv, key=lambda x: x.distance)

# Select the top 10% accurate matches

num_matches = int(len(matches_cv) * 0.1)  # choose 10% of total matches
best_matches = matches_cv[:num_matches]

pts1_best = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 2)
pts2_best = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 2)

inlier_sample = np.zeros((10,))
iter_arr = np.arange(2, 240, 30)
ransac = []

for iter in range(2, 240, 30):
    for i in range(10):
        H2to1, inliers = computeH_ransac(pts2_best, pts1_best, iter, inlier_tol = 2, adaptive = True)
        num_inliers = np.sum(inliers)
        inlier_sample[i] = num_inliers

    inlier_avg = np.mean(inlier_sample)
    ransac.append(inlier_avg)


# visualize the number of inliers for it
plt.plot(iter_arr, ransac, marker='s', color = '#147062',linewidth = 3)
plt.title('The number of inliers per RANSAC iteration', pad=15)
plt.xlabel('Iteration number', labelpad=10)
plt.ylabel('Inlier number', labelpad=10)
plt.show()