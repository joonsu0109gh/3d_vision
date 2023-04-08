import numpy as np
import cv2
from matchPics import matchPics


#Q6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
histogram = np.zeros(36)

#Match features for rotated images and update histogram
for i in range(0,360,10):
    #Rotate image
    rotated_cv_cover = scipy.ndimage.rotate(cv_cover, i)
    #Compute features, descriptors and match features
    matches, locs1, locs2 = matchPics(rotated_cv_cover, cv_cover)
    
    #Update histogram
    for r in range(len(matches)):
        histogram.append(i)
    
    #Display feature matching result for specific orientations
    if i == 10 or i == 70 or i == 190:
        plotMatches(cv_cover, rotated_cv_cover, matches, locs1, locs2)
        
# Display histogram
plt.hist(histogram,
         color = '#147062',
         edgecolor = 'black',
         linewidth = 1,
         bins=36)
plt.title('BRIEF and Rotations')
plt.xlabel('Degree')
plt.ylabel('Matching keypoints')
plt.show()