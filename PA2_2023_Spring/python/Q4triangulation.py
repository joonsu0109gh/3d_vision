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

        """
        Write your own code here 

        Step 1. Compute the camera projection matrices P1 using self.K1
        P1

        Step 2. Use hlp.camera2 to get 4 camera projection matrices P2
        P2s

        for loop range of 4:
        
            Step 3. Run triangulate using the projection matrices
            pts3d = self.triangulate(P1, self.pts1, P2, self.pts2)

            Step 4. Figure out the correct P2
            P2

        """

        # DO NOT CHANGE HERE!
        self.M1, self.M2 = M1, M2
        self.P1, self.P2 = P1, P2
        self.pts3d = pts3d
        

    """
    Q4 Triangulation
        [I] P1, camera projection matrix 1 (3x4 matrix)
            pts1, points in image 1 (Nx2 matrix)
            P2, camera projection matrix 2 (3x4 matrix)
            pts2, points in image 2 (Nx2 matrix)
        [O] pts3d, 3D points in space (Nx3 matrix)
    """
    def triangulate(self, P1, pts1, P2, pts2):
        """
        Write your own code here 


        

        replace pass by your implementation
        """
        pass
    
    


if __name__ == "__main__":

    Q4 = Q4()
    print("M2=", Q4.M2)







