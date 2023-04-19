import numpy as np
import cv2
#Import necessary functions
from matplotlib import pyplot as plt
import skimage.io
import skimage.color
import sys
#Import necessary functions
sys.path.append('../python')

import planarH, my_functions
from planarH import computeH_ransac, compositeH, computeH_norm
from my_functions import  compositeH_panorama, compositeH_panorama_blend, find_location_relationship, panorama_stitching, find_destination_img
from itertools import permutations

#Write script for Q14
img1 = cv2.imread('../data/image1.png')
img2 = cv2.imread('../data/image2.png')
img3 = cv2.imread('../data/image3.png')
img4 = cv2.imread('../data/image4.png')


#Write script for Q14
img_dict = {}
img_dict['img1'] = img1
img_dict['img2'] = img2
img_dict['img3'] = img3
img_dict['img4'] = img4
# First, find the order of the given images
imgs_dict = find_destination_img(img_dict)

dict_for_counting = {}

for i in imgs_dict.keys():
    dict_for_counting[i] = len(set(imgs_dict[i].values()))

max_value = max(dict_for_counting.values())
for key, value in dict_for_counting.items():
    if value == max_value:
        center_img = key

#center_img = find_key_with_largest_value(dict_for_counting)
print("Center img is", center_img)
for i, j in imgs_dict[center_img].items():
    print(i, "is on", j)
    
# Create a panorama with two neighbor images and save the panorama
count = 2
for i in imgs_dict[center_img].keys():
    if count == 2:
        panorama = img_dict.get(center_img)
    panorama = panorama_stitching(img_dict.get(i), panorama)
    cv2.imwrite(f'../result/panorama_{count}.png', panorama)
    count += 1
