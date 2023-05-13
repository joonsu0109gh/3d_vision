import numpy as np
import cv2 as cv 
import helper as hlp
import skimage.io as io
import matplotlib.pyplot as plt
import numpy.linalg as la
from Q2eppipolar_correspodences import Q2

# Q3

class Q3(Q2):
    def __init__(self):
        super(Q3, self).__init__()

        """
        Write your own code here 

        Step 1. Load intrinsic matrices from data/intrinsics.npz
        K1, K2

        Step 2. Run epipolar_correspondences to get points in image 2
        E = self.essential_matrix(self.F, K1, K2)

        """

        # DO NOT CHANGE HERE!
        self.K1 = K1
        self.K2 = K2
        self.E = E


    """
    Q3 Essential Matrix
        [I] F, the fundamental matrix (3x3 matrix)
            K1, camera matrix 1 (3x3 matrix)
            K2, camera matrix 2 (3x3 matrix)
        [O] E, the essential matrix (3x3 matrix)
    """
    def essential_matrix(self, F, K1, K2):
        """
        Write your own code here 


        

        replace pass by your implementation
        """
        pass



if __name__ == "__main__":

    Q3 = Q3()
    print("Essential Matrix : ", Q3.E)


