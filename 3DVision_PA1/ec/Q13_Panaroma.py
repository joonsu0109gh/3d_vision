import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import skimage.io
import skimage.color

#Import necessary functions
sys.path.append('../python')
import planarH
from planarH import computeH_ransac, compositeH, computeH_norm, compositeH_panorama, compositeH_panorama_blend


#Write script for Q13
img_dst = cv2.imread('../data/pano_left_hand.jpg')
img_src = cv2.imread('../data/pano_right_hand.jpg')
template_diagonal = int((img_src.shape[1]**2 + img_src.shape[0]**2)**(1/2))
img_dst_padded = cv2.copyMakeBorder(img_dst, top = template_diagonal, bottom= template_diagonal, left = template_diagonal, right = template_diagonal, borderType = 0)

# Load images
# Detect keypoints and extract descriptors
orb = cv2.ORB_create(nfeatures=10000)

kp1, des1 = orb.detectAndCompute(img_src, None)
kp2, des2 = orb.detectAndCompute(img_dst_padded, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches_cv = bf.match(des1, des2)

# Sort matches by distance
matches_cv = sorted(matches_cv, key=lambda x: x.distance)

# Select the top 10% accurate matches

num_matches = int(len(matches_cv) * 0.1)  # choose 10% of total matches
best_matches = matches_cv[:num_matches]

pts1_best = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 2)
pts2_best = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 2)

H2to1, inliers = computeH_ransac(pts2_best, pts1_best, 300, 2)

H2to1 = computeH_norm(pts2_best[inliers], pts1_best[inliers])

panorama = compositeH_panorama(H2to1, img_dst, img_src)
result = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(30,10))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB))
plt.title('Left image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))
plt.title('Right image')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(result)
plt.title('Stitched image')
plt.axis('off')

plt.show()
