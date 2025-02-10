import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


GAUSSIAN_SIZE = 7

# Initialise numDisparities, ranging from [16, 512], increased by 16 per step
numDisparities = 16
# Initialise blockSize, ranging from [5, 125], increase by 2 per step
blockSize = 5

# Initialize default threshold values in Canny
low_threshold = 10
high_threshold = 50

# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================

def onChangeNumDisparities(val):
    global numDisparities

    numDisparities = 16 + 16 * val

    updateDisparity()

def onChangeBlockSize(val):
    global blockSize
    
    blockSize = 5 + 2 * val

    updateDisparity()

# Update disparity map and display it
def updateDisparity():
    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    cv2.imshow('Disparity', disparityImg)


def getBlurredEdgesImage(image, low_T, high_T, blur_size):
    blurred_image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    edges = cv2.Canny(blurred_image, low_T, high_T)

    return edges

def onChangeLowThreshold(val):
    global low_threshold

    low_threshold = val

    updateEdges()

def onChangeHighThreshold(val):
    global high_threshold

    high_threshold = val

    updateEdges()

def updateEdges():
    edges = getBlurredEdgesImage(imgL, low_threshold, high_threshold, GAUSSIAN_SIZE)

    cv2.imshow('Edge Image', edges)

# ================================================
#
if __name__ == '__main__':

    # Load left image
    filename = 'umbrella_edgesL_7x7_7-54.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # cv2.imshow('Left Image', imgL)

    # Load right image
    filename = 'umbrella_edgesR_7x7_7-54.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # cv2.imshow('Right Image', imgR)


    # # Choose Canny edge parameters:
    # cv2.namedWindow('Edge Image')
    # cv2.createTrackbar('Low Threshold', 'Edge Image', low_threshold, 255, onChangeLowThreshold)
    # cv2.createTrackbar('High Threshold', 'Edge Image', high_threshold, 255, onChangeHighThreshold)

    # # Initial display
    # updateEdges()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Create a window and trackbars for the two parameters
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('numDisparities Step', 'Disparity', 0, 31, onChangeNumDisparities)
    cv2.createTrackbar('blockSize Step', 'Disparity', 0, 60, onChangeBlockSize)

    # Initialise the disparity image with default parameters
    updateDisparity()


    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()