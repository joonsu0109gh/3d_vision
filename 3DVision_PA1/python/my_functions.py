import numpy as np
import cv2
from matplotlib import pyplot as plt
from planarH import computeH_ransac, computeH_norm

def compositeH_panorama_blend(H2to1, dst, src):
    # src = cover
    # dst = desk
    # Create mask of same size as template
    src_diagonal = int((src.shape[1]**2 + src.shape[0]**2)**(1/2))
    src_mask = np.full_like(src, 1)
    dst_mask = np.full_like(dst, 0)
    dst_mask[dst != 0] = 1
    
    dst_padded = cv2.copyMakeBorder(dst, top = src_diagonal, bottom= src_diagonal, left = src_diagonal, right = src_diagonal, borderType = 0)
    dst_mask_padded = cv2.copyMakeBorder(dst_mask, top = src_diagonal, bottom= src_diagonal, left = src_diagonal, right = src_diagonal, borderType = 0)

    
    # Warp template by appropriate homography
    warped_src = cv2.warpPerspective(src, H2to1, (dst_padded.shape[1], dst_padded.shape[0]))
    warped_src_mask = cv2.warpPerspective(src_mask, H2to1, (dst_padded.shape[1], dst_padded.shape[0]))

    panorama_mask = warped_src_mask + dst_mask_padded

    panorama = cv2.addWeighted(src1 = dst_padded, alpha = 0.5, src2 = warped_src, beta = 0.5, gamma = 0)

    panorama[panorama_mask == 2] = panorama[panorama_mask == 2] // 2

    panorama = panorama * 2

    # trim the empty space of an image
    trimmed_panorama = remove_padding(panorama)

    # Extract the trimmed region from the original image
    #trimmed_panorama = panorama[y:y+h, x:x+w]

    return trimmed_panorama


def compositeH_panorama(H2to1, dst, src):
    # src = cover
    # dst = desk
    # Create mask of same size as template
    src_diagonal = int((src.shape[1]**2 + src.shape[0]**2)**(1/2))
    src_mask = np.full_like(src, 1)
    dst_mask = np.full_like(dst, 0)
    dst_mask[dst != 0] = 1
    
    dst_padded = cv2.copyMakeBorder(dst, top = src_diagonal, bottom= src_diagonal, left = src_diagonal, right = src_diagonal, borderType = 0)
    dst_mask_padded = cv2.copyMakeBorder(dst_mask, top = src_diagonal, bottom= src_diagonal, left = src_diagonal, right = src_diagonal, borderType = 0)

    # Warp template by appropriate homography
    warped_src_mask = cv2.warpPerspective(src_mask, H2to1, (dst_padded.shape[1], dst_padded.shape[0]))

    warped_src = cv2.warpPerspective(src, H2to1, (dst_padded.shape[1], dst_padded.shape[0]))

    dst_padded[warped_src_mask == 1] = warped_src[warped_src_mask == 1]
    #warped_src[dst_mask_padded == 1] = dst_padded[dst_mask_padded == 1]

    panorama = dst_padded
    #panorama = warped_src

    trimmed_panorama = remove_padding(panorama)
    return trimmed_panorama

def draw_matchings(img_src, img_dst):
    # Load images
    # Detect keypoints and extract descriptors
    orb = cv2.ORB_create(nfeatures=10000)

    kp1, des1 = orb.detectAndCompute(img_src, None)
    kp2, des2 = orb.detectAndCompute(img_dst, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_cv = bf.match(des1, des2)

    # Sort matches by distance
    matches_cv = sorted(matches_cv, key=lambda x: x.distance)

    # Select the top 10% accurate matches

    num_matches = int(len(matches_cv) * 0.1)  # choose 10% of total matches
    best_matches = matches_cv[:num_matches]
    img_matches = cv2.drawMatches(img_src, kp1,img_dst,  kp2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def get_H(img_src, img_dst):

    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=10000)

    kp1, des1 = orb.detectAndCompute(img_src, None)
    kp2, des2 = orb.detectAndCompute(img_dst, None)

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

    return H2to1

def find_location_relationship(img_src, img_dst):


    src_diagonal = int((img_src.shape[1]**2 + img_src.shape[0]**2)**(1/2))

    dst_padded = cv2.copyMakeBorder(img_dst, top = src_diagonal, bottom= src_diagonal, left = src_diagonal, right = src_diagonal, borderType = 0)

    H2to1 = get_H(img_src, dst_padded)

    src_1 = np.array([[0], [0], [1]])
    src_1_hom = H2to1 @ src_1
    src_1_hom = src_1_hom[:2, :] / (src_1_hom[2, :] + np.finfo(float).eps)

    src_2 = np.array([[0],[img_src.shape[0]], [1]])
    src_2_hom = H2to1 @ src_2
    src_2_hom = src_2_hom[:2, :] / (src_2_hom[2, :] + np.finfo(float).eps)

    src_3 = np.array([[img_src.shape[1]], [img_src.shape[0]], [1]])
    src_3_hom = H2to1 @ src_3
    src_3_hom = src_3_hom[:2, :] / (src_3_hom[2, :] + np.finfo(float).eps)

    src_4 = np.array([[img_src.shape[1]], [0], [1]])
    src_4_hom = H2to1 @ src_4
    src_4_hom = src_4_hom[:2, :] / (src_4_hom[2, :] + np.finfo(float).eps)

    src_center = ((src_1_hom[0][0] + src_2_hom[0][0] + src_3_hom[0][0] + src_4_hom[0][0])/4, (src_1_hom[1][0] + src_2_hom[1][0] + src_3_hom[1][0] + src_4_hom[1][0])/4)
    dst_center = (img_dst.shape[1]/2 + src_diagonal, img_dst.shape[0]/2 + src_diagonal)

    x_dist = dst_center[0]- src_center[0]
    y_dist = dst_center[1]- src_center[1]

    if (abs(x_dist) > abs(y_dist)) and (x_dist > 0):
        src_loc = "left"
    elif (abs(x_dist) > abs(y_dist)) and (x_dist < 0):
        src_loc = "right"
    elif (abs(x_dist) < abs(y_dist)) and (y_dist > 0):
        src_loc = "top"
    elif (abs(x_dist) < abs(y_dist)) and (x_dist < 0):
        src_loc = "bottom"
    else:
        src_loc = "whatever"

    return src_loc

def find_destination_img(img_dict):
    
    img_loc_dict = {}

    loc_nto1 = {}
    loc_nto1["img2"] = find_location_relationship(img_dict["img2"], img_dict["img1"])
    loc_nto1["img3"] = find_location_relationship(img_dict["img3"], img_dict["img1"])
    loc_nto1["img4"] = find_location_relationship(img_dict["img4"], img_dict["img1"])
    img_loc_dict["img1"] = loc_nto1
    print("location imgN to img1", loc_nto1)

    loc_nto2 = {}
    loc_nto2["img1"] = find_location_relationship(img_dict["img1"], img_dict["img2"])
    loc_nto2["img3"] = find_location_relationship(img_dict["img3"], img_dict["img2"])
    loc_nto2["img4"] = find_location_relationship(img_dict["img4"], img_dict["img2"])
    img_loc_dict["img2"] = loc_nto2
    print("location imgN to img2", loc_nto2)

    loc_nto3 = {}
    loc_nto3["img1"] = find_location_relationship(img_dict["img1"], img_dict["img3"])
    loc_nto3["img2"] = find_location_relationship(img_dict["img2"], img_dict["img3"])
    loc_nto3["img4"] = find_location_relationship(img_dict["img4"], img_dict["img3"])
    img_loc_dict["img3"] = loc_nto3
    print("location imgN to img3", loc_nto3)

    loc_nto4 = {}
    loc_nto4["img1"] = find_location_relationship(img_dict["img1"], img_dict["img4"])
    loc_nto4["img2"] = find_location_relationship(img_dict["img2"], img_dict["img4"])
    loc_nto4["img3"] = find_location_relationship(img_dict["img3"], img_dict["img4"])
    img_loc_dict["img4"] = loc_nto4
    print("location imgN to img4", loc_nto4)

    return img_loc_dict

def panorama_stitching(img_src, img_dst):
    template_diagonal = int((img_src.shape[1]**2 + img_src.shape[0]**2)**(1/2))
    img_dst_padded = cv2.copyMakeBorder(img_dst, top = template_diagonal, bottom= template_diagonal, left = template_diagonal, right = template_diagonal, borderType = 0)

    H2to1 = get_H(img_src, img_dst_padded)

    panorama = compositeH_panorama(H2to1, img_dst, img_src)

    return panorama

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

def remove_padding(image):
    """Remove padding from an image.

    Args:
        image (numpy.ndarray): The padded image to remove padding from.

    Returns:
        numpy.ndarray: The original image with padding removed.

    """
    # Find the padding size by looking at the first row and column of the image
    top_pad = 0
    while np.all(image[top_pad, :] == image[0, :]):
        top_pad += 1
    bottom_pad = 0
    while np.all(image[-bottom_pad - 1, :] == image[-1, :]):
        bottom_pad += 1
    left_pad = 0
    while np.all(image[:, left_pad] == image[:, 0]):
        left_pad += 1
    right_pad = 0
    while np.all(image[:, -right_pad - 1] == image[:, -1]):
        right_pad += 1

    # Crop the image to remove the padding
    return image[top_pad:-bottom_pad, left_pad:-right_pad]