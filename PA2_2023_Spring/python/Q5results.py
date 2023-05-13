import numpy as np
import cv2 as cv 
import helper as hlp
import skimage.io as io
import numpy.linalg as la
import matplotlib.pyplot as plt
from Q4triangulation import Q4

# Q5

class Q5(Q4):
    def __init__(self):
        super(Q5, self).__init__()

        # Step 1. Scatter plot the correct 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pts3d[:, 0], self.pts3d[:, 1], self.pts3d[:, 2], c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Point Cloud')
        plt.show()

        # Step 2. Load the points from data/some_corresp.npz
        corresp = np.load('../data/some_corresp.npz')
        pts1 = corresp['pts1']
        pts2 = corresp['pts2']

        # Step 3. Calculate the reprojection error
        pts3d = self.triangulate(self.P1, pts1, self.P2, pts2)
        reproj_err = self.compute_reprojerr(self.P1, pts1, self.P2, pts2, pts3d)
        print("Reprojection Error:", reproj_err)

        # Step 4. Save the computed extrinsic parameters to data/extrinsics.npz
        extrinsics = {'R1': self.M1[:, :3], 'R2': self.M2[:, :3], 't1': self.M1[:, 3], 't2': self.M2[:, 3]}
        np.savez('../data/extrinsics.npz', **extrinsics)
        

    """
    Q5 Compute Reprojection Error
        [I] P1, camera projection matrix 1 (3x4 matrix)
            pts1, points in image 1 (Nx2 matrix)
            P2, camera projection matrix 2 (3x4 matrix)
            pts2, points in image 2 (Nx2 matrix)
            pts3d, 3D points in space (Nx3 matrix)
        [O] reproj_err, Reprojection Error (float)
    """
    def compute_reprojerr(self, P1, pts1, P2, pts2, pts3d):
        pts3d_homog = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))

        # Project 3D points to image coordinates
        projected_pts = P1 @ pts3d_homog.T
        projected_pts1 = projected_pts[:2] / projected_pts[2]

        # Calculate Euclidean distance between projected points and actual points in image 1
        diff = pts1 - projected_pts1
        reproj_err = np.mean(np.linalg.norm(diff, axis=1))

        return reproj_err


if __name__ == "__main__":

    Q5 = Q5()
