import cv2 as cv
import numpy as np
import helper as hlp
import numpy.linalg as la
import skimage.color as col
import matplotlib.pyplot as plt
from Q7disparity import Q7

class Q8(Q7):
    # DO NOT CHANGE HERE!
    def __init__(self):
        super(Q8, self).__init__()
        # Load the rectify parameters.
        rectify = np.load('../data/rectify.npz')
        M1, M2 = rectify['M1'], rectify['M2']
        K1p, K2p = rectify['K1p'], rectify['K2p']
        R1p, R2p = rectify['R1p'], rectify['R2p']
        t1p, t2p = rectify['t1p'], rectify['t2p']

        depth = self.get_depth(self.disp, K1p, K2p, R1p, R2p, t1p, t2p)
        depth = np.where(self.I1<=40, np.inf, depth)
        self.depth = depth

    
    """
    Q8 Depth Map
        [I] dispM, disparity map (H1xW1 matrix)
            K1 K2, camera matrices (3x3 matrix)
            R1 R2, rotation matrices (3x3 matrix)
            t1 t2, translation vectors (3x1 matrix)
        [O] depthM, depth map (H1xW1 matrix)
    """
    def get_depth(self, dispM, K1p, K2p, R1p, R2p, t1p, t2p):
        
        """
        Write your own code here 


        

        replace pass by your implementation
        """
        pass


if __name__ == "__main__":

    Q8 = Q8()

    # Display disparity and depth maps
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(Q8.disp, cmap='inferno')
    ax1.set_title('Disparity Image')
    ax1.set_axis_off()
    ax2.imshow(Q8.depth, cmap='inferno')
    ax2.set_title('Depth Image')
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()



