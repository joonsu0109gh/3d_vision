import numpy as np
import cv2 as cv 
import helper as hlp
import skimage.io as io
import matplotlib.pyplot as plt
import numpy.linalg as la
from Q1eight_point import Q1

# Q2

class Q2(Q1):
    def __init__(self):
        super(Q2, self).__init__()

        """
        Write your own code here 

        Step 1. Load points in image 1 from data/temple_coords.npz
        pts1 

        Step 2. Run epipolar_correspondences to get points in image 2
        pts2 = self.epipolar_correspondences(self.im1, self.im2, self.F, pts1)

        """

        # DO NOT CHANGE HERE!
        self.pts1 = np.load('../data/temple_coords.npz')['pts1']
        self.pts2 = self.epipolar_correspondences(self.im1, self.im2, self.F, self.pts1)


    """
    Q2 Epipolar Correspondences
        [I] im1, image 1 (H1xW1 matrix)
            im2, image 2 (H2xW2 matrix)
            F, fundamental matrix from image 1 to image 2 (3x3 matrix)
            pts1, points in image 1 (Nx2 matrix)
        [O] pts2, points in image 2 (Nx2 matrix)
    """
    def epipolar_correspondences(self, im1, im2, F, pts1):
        window_size = 5
        num_points = pts1.shape[0]
        pts2 = np.zeros_like(pts1)

        for i in range(num_points):
            x, y = pts1[i]
            v = np.array([[x], [y], [1]])
            l = F @ v
            a, b, c = l[0], l[1], l[2]

            # Generate candidate points along the epipolar line
            min_idx = max(0, int(x) - window_size)
            max_idx = min(im2.shape[1] - 1, int(x) + window_size)
            candidate_points = np.arange(min_idx, max_idx + 1)

            # Compute similarity scores between target window and candidate windows
            scores = []
            for candidate_x in candidate_points:
                window1 = im1[y - window_size: y + window_size + 1, x - window_size: x + window_size + 1]
                window2 = im2[y - window_size: y + window_size + 1, candidate_x - window_size: candidate_x + window_size + 1]
                score = np.sum(np.abs(window1 - window2))
                scores.append(score)

            # Find the candidate point with the highest score
            best_candidate_idx = np.argmin(scores)
            x_candidate = candidate_points[best_candidate_idx]
            pts2[i] = [x_candidate, y]

        return pts2


    def epipolarMatchGUI(self, I1, I2, F):
        sy, sx, sd = I2.shape
        f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
        ax1.imshow(I1)
        ax1.set_title('Select a point in this image')
        ax1.set_axis_off()
        ax2.imshow(I2)
        ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
        ax2.set_axis_off()
        while True:
            plt.sca(ax1)
            x, y = plt.ginput(1, mouse_stop=2)[0]
            xc, yc = int(x), int(y)
            v = np.array([[xc], [yc], [1]])
            l = F @ v
            s = np.sqrt(l[0]**2+l[1]**2)
            if s==0:
                hlp.error('Zero line vector in displayEpipolar')
            l = l / s
            if l[0] != 0:
                xs = 0
                xe = sx - 1
                ys = -(l[0] * xs + l[2]) / l[1]
                ye = -(l[0] * xe + l[2]) / l[1]
            else:
                ys = 0
                ye = sy - 1
                xs = -(l[1] * ys + l[2]) / l[0]
                xe = -(l[1] * ye + l[2]) / l[0]
            ax1.plot(x, y, '*', markersize=6, linewidth=2)
            ax2.plot([xs, xe], [ys, ye], linewidth=2)
            # draw points
            pc = np.array([[xc, yc]])
            p2 = self.epipolar_correspondences(I1, I2, F, pc)
            ax2.plot(p2[0,0], p2[0,1], 'ro', markersize=8, linewidth=2)
            plt.draw()



if __name__ == "__main__":

    Q2 = Q2()
    Q2.epipolarMatchGUI(Q2.im1, Q2.im1, Q2.F)



