import numpy as np
import cv2
from matplotlib import pyplot as plt

def computeH(x1, x2):
    """
    Computes the homography that maps from x2 to x1
    :param x1: numpy array of shape (n, 2), representing the coordinates of n feature points in image 1
    :param x2: numpy array of shape (n, 2), representing the coordinates of the corresponding feature points in image 2
    :return H2to1: numpy array of shape (3, 3), representing the homography matrix from image 2 to image 1
    """

    # Step 1: Construct the A matrix
    n = x1.shape[0]
    
    A = np.zeros((2 * n, 9)) 
    for i in range(n):
        
        A[2 * i, :] = [-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x2[i, 0] * x1[i, 0], x2[i, 1] * x1[i, 0], x1[i, 0]]
        # Fill in the second row of A with zeros, the x and y coordinates of x2[i], a 1, and more zeros
        A[2 * i + 1, :] = [0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x2[i, 0] * x1[i, 1], x2[i, 1] * x1[i, 1], x1[i, 1]]
        
    # Step 2: Solve for H using SVD
    U, S, V = np.linalg.svd(A)
    H2to1 = V[-1, :].reshape((3, 3))
    return H2to1


def computeH_norm(x1, x2):
    #Q9
    # Compute the centroid of the points
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    centered_x1 = x1 - centroid1
    centered_x2 = x2 - centroid2

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_dist1 = np.max(np.sqrt(centered_x1[:, 0]**2 + centered_x1[:, 1]**2))
    max_dist2 = np.max(np.sqrt(centered_x2[:, 0]**2 + centered_x2[:, 1]**2))

    scale1 = np.sqrt(2) / max_dist1
    scale2 = np.sqrt(2) / max_dist2
    norm_x1 = centered_x1 * scale1
    norm_x2 = centered_x2 * scale2

    # Similarity transform 1
    T1 = np.array([[scale1, 0, -scale1 * centroid1[0]],
                   [0, scale1, -scale1 * centroid1[1]],
                   [0, 0, 1]])

    # Similarity transform 2
    T2 = np.array([[scale2, 0, -scale2 * centroid2[0]],
                   [0, scale2, -scale2 * centroid2[1]],
                   [0, 0, 1]])

    # Compute homography
    H2to1 = computeH(norm_x1, norm_x2)

    # Denormalization
    H2to1 = np.dot(np.linalg.inv(T1), np.dot(H2to1, T2))

    return H2to1



def computeH_ransac(locs1, locs2, num_iter=1500, inlier_tol=4):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    n = locs1.shape[0]
    max_inliers = -1
    bestH2to1 = None
    inliers = None
    
    for i in range(num_iter):
        # Randomly sample 4 points
        indices = np.random.choice(n, 4, replace=False)
        x1_sample = locs1[indices, :]
        x2_sample = locs2[indices, :]

        # Compute the homography using the sample points
        H_sample = computeH_norm(x1_sample, x2_sample)

        # Compute the projection error for all points
        homogeneous_locs2 = np.concatenate((locs2, np.ones((n, 1))), axis=1)
        locs2_hom = H_sample @ homogeneous_locs2.T  # convert to homography coordinates
        locs2_hom = locs2_hom[:2, :] / (locs2_hom[2, :] + np.finfo(float).eps)  # add eps to avoid division by zero
        errors = np.linalg.norm(locs1 - locs2_hom.T, axis=1)

        # Count inliers
        num_inliers = np.sum(errors < inlier_tol)

        # Update best homography if current sample produces more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            #x1_sample = locs1[inliers, :]
            #x2_sample = locs2[inliers, :]
            bestH2to1 = H_sample
            inliers = errors < inlier_tol

    return bestH2to1, inliers

def computeH_ransac_adaptive(locs1, locs2, num_iter, inlier_tol=4):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    n = locs1.shape[0]
    max_inliers = -1
    bestH2to1 = None
    inliers = None
    
    p = 0.99  # Desired probability of finding at least one good sample
    eps = 1e-5  # A small value to avoid division by zero
    for i in range(num_iter):
        # Randomly sample 4 points
        indices = np.random.choice(n, 4, replace=False)
        x1_sample = locs1[indices, :]
        x2_sample = locs2[indices, :]

        # Compute the homography using the sample points
        H_sample = computeH_norm(x1_sample, x2_sample)

        # Compute the projection error for all points
        homogeneous_locs2 = np.concatenate((locs2, np.ones((n, 1))), axis=1)
        locs2_hom = H_sample @ homogeneous_locs2.T  # convert to homography coordinates
        locs2_hom = locs2_hom[:2, :] / (locs2_hom[2, :] + np.finfo(float).eps)  # add eps to avoid division by zero
        errors = np.linalg.norm(locs1 - locs2_hom.T, axis=1)

        # Count inliers
        num_inliers = np.sum(errors < inlier_tol)

        # Update best homography if current sample produces more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            #x1_sample = locs1[inliers, :]
            #x2_sample = locs2[inliers, :]
            bestH2to1 = H_sample
            inliers = errors < inlier_tol
            
            
            # Update the number of iterations based on the current inlier ratio
            inlier_ratio = float(max_inliers) / n

            if inlier_ratio == 1:
                # If inlier_ratio is exactly 1, set it to a slightly smaller value
                inlier_ratio -= eps

            if 1 - (inlier_ratio + eps)**4 <= 0:
                # If the value inside the logarithm is negative or zero, set num_iter to a very large value
                num_iter = np.inf
            else:
                num_iter = int(np.log(1 - p) / np.log(1 - (inlier_ratio + eps)**4))

    return bestH2to1, inliers
'''

def compositeH(H2to1, template, img):
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Compute inverse homography
    H1to2 = np.linalg.inv(H2to1)
    
    # Create mask of same size as template
    mask = np.ones_like(img)

    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H1to2, (template.shape[1], template.shape[0]))

    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(img, H1to2, (template.shape[1], template.shape[0]))

    # Use mask to combine the warped template and the image
    composite_img = template.copy()
    composite_img[warped_mask == 1] = warped_template[warped_mask == 1]

    return composite_img
'''
def compositeH(H2to1, template, img):
    # template = harry
    # img = desk
    # Create mask of same size as template

    mask = np.full_like(template, 1)

    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))
    
    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

    # Use mask to combine the warped template and the image
    composite_img = img.copy()
    composite_img[warped_mask == 1] = warped_template[warped_mask == 1]

    return composite_img

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

    #panorama = panorama * 2

    # trim the empty space of an image
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    # Find the non-zero (foreground) region of the image
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the trimmed region from the original image
    trimmed_panorama = panorama[y:y+h, x:x+w]

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
    warped_src = cv2.warpPerspective(src, H2to1, (dst_padded.shape[1], dst_padded.shape[0]))

    warped_src[dst_mask_padded == 1] = dst_padded[dst_mask_padded == 1]

    panorama = warped_src

    # trim the empty space of an image
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    # Find the non-zero (foreground) region of the image
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the trimmed region from the original image
    trimmed_panorama = panorama[y:y+h, x:x+w]

    return trimmed_panorama


def draw_matchings(img_src, img_dst):
    # Load images
    # Detect keypoints and extract descriptors
    orb = cv2.ORB_create(nfeatures=500,
                        scaleFactor=1.2,
                        nlevels=8,
                        edgeThreshold=31,
                        firstLevel=0,
                        WTA_K=2,
                        scoreType=cv2.ORB_HARRIS_SCORE,
                        patchSize=31,
                        fastThreshold=20)

    kp1, des1 = orb.detectAndCompute(img_src, None)
    kp2, des2 = orb.detectAndCompute(img_dst, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_cv = bf.match(des1, des2)

    # Sort matches by distance
    matches_cv = sorted(matches_cv, key=lambda x: x.distance)

    # Select the top 10% accurate matches

    num_matches = int(len(matches_cv) * 0.01)  # choose 10% of total matches
    best_matches = matches_cv[:num_matches]
    img_matches = cv2.drawMatches(img_src, kp1,img_dst,  kp2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the image with matches
    plt.imshow(img_matches)
    plt.axis('off')
    plt.show()

def get_H(img_src, img_dst):
    orb = cv2.ORB_create(nfeatures=1000,
                        scaleFactor=1.2,
                        nlevels=8,
                        edgeThreshold=31,
                        firstLevel=0,
                        WTA_K=2,
                        scoreType=cv2.ORB_HARRIS_SCORE,
                        patchSize=31,
                        fastThreshold=20)

    kp1, des1 = orb.detectAndCompute(img_src, None)
    kp2, des2 = orb.detectAndCompute(img_dst, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_cv = bf.match(des1, des2)

    # Sort matches by distance
    matches_cv = sorted(matches_cv, key=lambda x: x.distance)

    # Select the top 10% accurate matches

    num_matches = int(len(matches_cv) * 0.5)  # choose 10% of total matches
    best_matches = matches_cv[:num_matches]
    pts1_best = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 2)
    pts2_best = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 2)

    H2to1, inliers = computeH_ransac_adaptive(pts2_best, pts1_best, 300, 2)

    H2to1 = computeH_norm(pts2_best[inliers], pts1_best[inliers])

    panorama = compositeH_panorama(H2to1, img_dst, img_src)
    result = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.axis('off')
    plt.show()