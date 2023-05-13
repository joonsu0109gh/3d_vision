import cv2 as cv
import numpy as np
import helper as hlp
import numpy.linalg as la
import skimage.color as col
import matplotlib.pyplot as plt
from Q6rectify import Q6

class Q7(Q6):
    # DO NOT CHANGE HERE!
    def __init__(self):
        super(Q7, self).__init__()
        # Load the images rectified at Q6.
        I1 = self.I1
        I2 = self.I2

        # Get disparity and depth maps
        min_disp, num_disp, win_size = 370, 60, 11
        dispM = self.get_disparity(I1, I2, min_disp, num_disp, win_size)

        dispI = np.where(dispM<min_disp, min_disp-1, dispM)
        dispI = np.where(dispI>=min_disp+num_disp, min_disp-1, dispI)
        dispI = np.where(I1<=40, np.inf, dispI)
        self.disp = dispI

        
    
    """
    Q7 Disparity Map
        [I] im1, image 1 (H1xW1 matrix)
            im2, image 2 (H2xW2 matrix)
            max_disp, scalar maximum disparity value
            win_size, scalar window size value
        [O] dispM, disparity map (H1xW1 matrix)
    """
    def get_disparity(self, I1, I2, min_disp, num_disp, win_size):
        
        """
        Write your own code here 


        

        replace pass by your implementation
        """
        pass


if __name__ == "__main__":

    Q7 = Q7()

    fig, (ax1) = plt.subplots(1, 1)
    ax1.imshow(Q7.disp, cmap='inferno')
    ax1.set_title('Disparity Image')
    ax1.set_axis_off()
    plt.tight_layout()
    plt.show()



