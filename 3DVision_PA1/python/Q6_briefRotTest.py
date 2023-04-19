import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
import scipy
from matplotlib import pyplot as plt


# Q6
cv_cover = cv2.imread('../data/cv_cover.jpg')

histogram = []
iter = 0

for i in range(0,360,10):
    # Rotate image
    rotated_cv_cover = scipy.ndimage.rotate(cv_cover, i)
    matches, locs1, locs2 = matchPics(rotated_cv_cover, cv_cover)

    # Update histogram
    for r in range(len(matches)):
        histogram.append(i)
    
    if i == 10 or i == 70 or i == 190:
        plotMatches(cv_cover, rotated_cv_cover, matches, locs1, locs2)


plt.hist(histogram
         ,color = '#147062'
         ,edgecolor = 'black'
         ,linewidth = 1
         , bins=36)

plt.title('BRIEF and Rotations')
plt.xlabel('Degree', labelpad=10)
plt.ylabel('Matching keypoints', labelpad=10)
plt.show()