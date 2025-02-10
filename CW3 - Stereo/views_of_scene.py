import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


NUM_DISPARITIES = 64
BLOCK_SIZE = 7

F_PIXELS = 5806.559
C_X = 1429.219
C_Y = 993.403
DOFFS = 114.291
BASELINE = 174.019


# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================

# ================================================
#
def plot(disparity, f_pixels, baseline, doffs, cx, cy):
    x = []
    y = []
    z = []

    height, width = disparity.shape
    min_disparity = np.min(disparity)

    # Calculate pixel by pixel
    for v in range(height):
        for u in range(width):
            d_i = disparity[v, u]    
            
            # Skip if the pixel is from the region that left and right images do not overlap
            if d_i > min_disparity:
                Z_i = baseline * f_pixels / (d_i + doffs)

                # X_i = (u - cx) * Z_i / f_pixels
                X_i = u * Z_i / f_pixels

                # Y_i = (v - cy) * Z_i / f_pixels
                Y_i = v * Z_i / f_pixels

                x.append(X_i)
                y.append(Y_i)
                z.append(Z_i)

    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter(x, z, y, 'green', s=0.2)

    # 3D view
    ax.view_init(elev=25, azim=65)
    
    # Top view
    # ax.view_init(elev=90, azim=90)
    
    # Side view
    # ax.view_init(elev=180, azim=0)

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    plt.savefig('myplot.png', bbox_inches='tight')
    plt.show()


# Load left image
filename = 'umbrella_edgesL_7x7_7-54.png'
imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#
if imgL is None:
    print('\nError: failed to open {}.\n'.format(filename))
    sys.exit()

# Load right image
filename = 'umbrella_edgesR_7x7_7-54.png'
imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#
if imgR is None:
    print('\nError: failed to open {}.\n'.format(filename))
    sys.exit()


# Get disparity map
disparity = getDisparityMap(imgL, imgR, NUM_DISPARITIES, BLOCK_SIZE)

# Show 3D plot of the scene
plot(disparity, F_PIXELS, BASELINE, DOFFS, C_X, C_Y)