"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import cv2
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE, imReadAndConvert
from ex1_utils import *

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    def gammaChangeFunction(gamma):
        gamma = cv2.getTrackbarPos('gamma', 'gammaDisplay')
        img=0
        if (gamma != 0):
            img = pow(imgOrig, (1.0 / (gamma/ 100)));
        elif (gamma == 0):
            img = pow(imgOrig, 1000.0 );
        cv2.imshow('gammaDisplay', img)
        k = cv2.waitKey(1)
    pass

    # Create a window

    if rep == LOAD_GRAY_SCALE:
        imgOrig = cv2.imread(img_path, 0)
    else:
        imgOrig = cv2.imread(img_path)
    imgOrig = normalize(imgOrig)

    cv2.namedWindow('gammaDisplay')
    h, w = imgOrig.shape[:2]
    cv2.resizeWindow("gammaDisplay", h, w);
    # create trackbars for color change
    cv2.createTrackbar('gamma', 'gammaDisplay', 100, 200, gammaChangeFunction)
    gammaChangeFunction(0)
    # Wait until user press some key
    cv2.waitKey()
    pass

def main():
    gammaDisplay('beach.jpg', LOAD_RGB)

if __name__ == '__main__':
    main()
