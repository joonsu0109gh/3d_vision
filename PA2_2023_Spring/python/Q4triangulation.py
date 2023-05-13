import numpy as np
import cv2 as cv 
import helper as hlp
import skimage.io as io
import matplotlib.pyplot as plt
import numpy.linalg as la
from Q3essential_matrix import Q3

# Q4

class Q4(Q3):
    def __init__(self):
        super(Q4, self).__init__()


        # Step 1. Compute the camera projection matrices P1 using self.K1
        P1 = np.dot(self.K1, np.hstack((np.eye(3), np.zeros((3,1)))))

        # Step 2. Use hlp.camera2 to get 4 camera projection matrices P2
        P2s = hlp.camera2(self.E)
        max_positive_depth = 0
        selected_P2 = None
        selected_pts3d = None

        for i in range(4):
            # Step 3. Run triangulate using the projection matrices
            P2 = P2s[:, :, i]
            # P2 = P2s[i, :, :]

            pts3d = self.triangulate(P1, self.pts1, P2, self.pts2)

            # Step 4. Figure out the correct P2
            num_positive_depth = np.sum(pts3d[:, 2] > 0)

            if num_positive_depth > max_positive_depth:
                max_positive_depth = num_positive_depth
                selected_P2 = P2
                selected_pts3d = pts3d
        
        # Store the results
        self.M1, self.M2 = P1, selected_P2
        self.P1, self.P2 = P1, selected_P2
        self.pts3d = selected_pts3d


    """
    Q4 Triangulation
        [I] P1, camera projection matrix 1 (3x4 matrix)
            pts1, points in image 1 (Nx2 matrix)
            P2, camera projection matrix 2 (3x4 matrix)
            pts2, points in image 2 (Nx2 matrix)
        [O] pts3d, 3D points in space (Nx3 matrix)
    """
    def triangulate(self, P1, pts1, P2, pts2):

        # Convert the points to homogeneous coordinates
        pts1_homog = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        pts2_homog = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

        # Transpose the points for cv.triangulatePoints
        pts1_homog = pts1_homog.T
        pts2_homog = pts2_homog.T

        # Adjust the dimensions of P1 and P2
        # P1 = np.vstack((P1, np.zeros((1, 4))))
        # P2 = np.vstack((P2, np.zeros((1, 4))))

        # P1z = P1[:3]
        # P2z = P2[:3]

        # Triangulate the points
        pts4d_homog = cv.triangulatePoints(P1, P2.T, pts1_homog[:2, :], pts2_homog[:2, :])

        # Convert points to non-homogeneous coordinates
        pts3d = (pts4d_homog[:3, :] / pts4d_homog[3, :]).T

        # Return the 3D points
        return pts3d



if __name__ == "__main__":

    Q4 = Q4()
    print("M2=", Q4.M2)
