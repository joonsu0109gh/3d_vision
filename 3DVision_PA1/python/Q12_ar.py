import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from planarH import computeH_ransac, compositeH, computeH_norm
from matchPics import matchPics
from matplotlib import pyplot as plt

from tqdm import tqdm
import time

import imageio

#Write script for Q12
def loadVid(filename):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    frames = [] 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames

cv_cover = cv2.imread('../data/cv_cover.jpg') # (440, 350, 3)
book_frames = loadVid("../data/book.mov") # (480, 640) 641 frame FPS 30
ar_frames = loadVid("../data/ar_source.mov") # (360, 640) 511 frame FPS 25

vid_frames = []

for book_frame, ar_frame in zip(book_frames, tqdm(ar_frames)):

    cropped_ar = ar_frame[44:316, 320-108 : 320+108]

    orb = cv2.ORB_create(nfeatures=10000)

    kp1, des1 = orb.detectAndCompute(cv_cover, None)
    kp2, des2 = orb.detectAndCompute(book_frame, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_cv = bf.match(des1, des2)

    # Sort matches by distance
    matches_cv = sorted(matches_cv, key=lambda x: x.distance)

    num_matches = int(len(matches_cv) * 0.5)  # choose 10% of total matches
    best_matches = matches_cv[:num_matches]

    pts1_best = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 2)
    pts2_best = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 2)

    H2to1, inliers = computeH_ransac(pts2_best, pts1_best, 300, 2, adaptive = True)
    
    H2to1 = computeH_norm(pts2_best[inliers], pts1_best[inliers])
    
    h, w = book_frames[0].shape[:2]

    cropped_ar = cv2.resize(cropped_ar, dsize=(cv_cover.shape[1], cv_cover.shape[0]), interpolation=cv2.INTER_AREA)

    warped_cropped_ar = cv2.warpPerspective(cropped_ar, H2to1, (w, h))

    result = compositeH(H2to1, cropped_ar, book_frame)

    vid_frames.append(result)

writer = imageio.get_writer('../result/ar_adaptive_O_0.5.avi', fps=25)

for i in range(len(vid_frames)):
    img = vid_frames[i]
    writer.append_data(img)

writer.close()
