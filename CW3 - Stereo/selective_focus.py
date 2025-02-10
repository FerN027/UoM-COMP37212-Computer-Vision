import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

# Kernel size for blurring the background
GAUSSIAN_SIZE = 21

# Ranging from [16, 512], increased by 16 per step
numDisparities = 16

# Ranging from [5, 125], increase by 2 per step
blockSize = 5

# Ranging from [0, 5], increased by 0.05 per step
k = 0

# Ranging from [0, 255], increased by 1 per step
threshold = 0


def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image

def onChangeNumDisparities(val):
    global numDisparities
    numDisparities = 16 + 16 * val

    updateAll()

def onChangeBlockSize(val):
    global blockSize
    blockSize = 5 + 2 * val

    updateAll()

def onChangeK(val):
    global k
    k = val / 20

    updateAll()

def onChangeThreshold(val):
    global threshold
    threshold = val

    updateAll()

def updateAll():
    # Get disparity map and depth image
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    depth = 1 / (disparity + k)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    depth_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply threshold
    _, mask = cv2.threshold(depth_normalized, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)

    # Blur the whole left greyscale image first
    blurred_image = cv2.GaussianBlur(imgL, (GAUSSIAN_SIZE, GAUSSIAN_SIZE), sigmaX=0, sigmaY=0)

    # Then use the mask to protect the object area
    final_image = np.where(mask == 255, blurred_image, imgL)
    final_image = final_image.astype('uint8')

    cv2.imshow('Disparity', disparityImg)
    cv2.imshow('Depth', depth_normalized)
    cv2.imshow('Mask', mask)
    cv2.imshow('Final Image', final_image)

# ==========================================================
# Load left image
filename = 'girlL.png'
imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#
if imgL is None:
    print('\nError: failed to open {}.\n'.format(filename))
    sys.exit()

# cv2.imshow('Left Image', imgL)

# Load right image
filename = 'girlR.png'
imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#
if imgR is None:
    print('\nError: failed to open {}.\n'.format(filename))
    sys.exit()

# cv2.imshow('Right Image', imgR)
# ==========================================================

# Create the disparity window
cv2.namedWindow('Disparity')
cv2.createTrackbar('numDisparities Step', 'Disparity', 0, 31, onChangeNumDisparities)
cv2.createTrackbar('blockSize Step', 'Disparity', 0, 60, onChangeBlockSize)

# Create the depth window
cv2.namedWindow('Depth')
cv2.createTrackbar('k * 20', 'Depth', 0, 100, onChangeK)

# Create the mask window
cv2.namedWindow('Mask')
cv2.createTrackbar('Threshold', 'Mask', 0, 255, onChangeThreshold)

# Create window for the final image
cv2.namedWindow('Final Image')

# Initialise the four windows
updateAll()


# Wait for spacebar press or escape before closing,
# otherwise window will close without you seeing it
while True:
    key = cv2.waitKey(1)
    if key == ord(' ') or key == 27:
        break

cv2.destroyAllWindows()