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
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 304976335


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: grayscale (1) or RGB(2)
    :return: The image object normalized
    """
    return normalize(imgRead(filename,representation)).astype(np.float)


def imgRead(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: grayscale (1) or RGB(2)
    :return: The image object
    """
    if representation==LOAD_GRAY_SCALE:
        img = cv2.imread(filename,0)
    else:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype('uint8')

def normalize(img):
    return (img/255).astype(np.float)

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: LOAD_GRAY_SCALE or RGB
    :return: None
    """
    img=imReadAndConvert(filename, representation)
    if representation==LOAD_GRAY_SCALE:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    YIQ_from_RGB = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    YIQImg = np.ndarray(imgRGB.shape)

    YIQImg[:, :, 0] = YIQ_from_RGB[0,0] * imgRGB[:, :, 0] + YIQ_from_RGB[0,1] * imgRGB[:, :, 1] + YIQ_from_RGB[0,2] * imgRGB[:, :, 2]
    YIQImg[:, :, 1] = YIQ_from_RGB[1,0] * imgRGB[:, :, 0] + YIQ_from_RGB[1,1] * imgRGB[:, :, 1] + YIQ_from_RGB[1,2] * imgRGB[:, :, 2]
    YIQImg[:, :, 2] = YIQ_from_RGB[2,0] * imgRGB[:, :, 0] + YIQ_from_RGB[2,1] * imgRGB[:, :, 1] + YIQ_from_RGB[2,2] * imgRGB[:, :, 2]

    return YIQImg

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    rgb_from_yiq = np.linalg.inv(yiq_from_rgb)

    RGBImg = np.ndarray(imgYIQ.shape)

    RGBImg[:, :, 0] = rgb_from_yiq[0,0] * imgYIQ[:, :, 0] + rgb_from_yiq[0,1] * imgYIQ[:, :, 1] + rgb_from_yiq[0,2] * imgYIQ[:, :, 2]
    RGBImg[:, :, 1] = rgb_from_yiq[1,0] * imgYIQ[:, :, 0] + rgb_from_yiq[1,1] * imgYIQ[:, :, 1] + rgb_from_yiq[1,2] * imgYIQ[:, :, 2]
    RGBImg[:, :, 2] = rgb_from_yiq[2,0] * imgYIQ[:, :, 0] + rgb_from_yiq[2,1] * imgYIQ[:, :, 1] + rgb_from_yiq[2,2] * imgYIQ[:, :, 2]

    return RGBImg

def unnormalize(img):
    return (img*255).astype('uint8')

def histogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """

    if isRGB(imgOrig):
        imgYIQ = transformRGB2YIQ(imgOrig)
        Y=imgYIQ[:,:,0]
        unnormImg = unnormalize(Y).astype('int')
    else:
        unnormImg = unnormalize(imgOrig).astype('int')

    histOrig = calHist(unnormImg)
    cumSumOrig = calCumSum(histOrig)

    h, w = unnormImg.shape[:2]
    LUT=(np.ceil(cumSumOrig*255/(h*w))).astype('uint8')
    imgEq=np.zeros_like(unnormImg)
    for i in range(h):
        for j in range (w):
                imgEq[i,j]=LUT[unnormImg[i,j]]
    histEq=calHist(imgEq)
    imgEq = normalize(imgEq)

    if isRGB(imgOrig):
        imgYIQ[:,:,0] = imgEq
        imgEq = transformYIQ2RGB(imgYIQ)

    return imgEq,histOrig,histEq

def isRGB(img: np.ndarray) -> bool:
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            return True
    return False

def calHist(img: np.ndarray) -> np.ndarray:
    img_flat = img.ravel()
    hist = np.zeros(256)

    for pix in img_flat:
        hist[pix] += 1

    return hist


def calCumSum(arr: np.array) -> np.ndarray:
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    arr_len = len(arr)

    for idx in range(1, arr_len):
       cum_sum[idx] = arr[idx] + cum_sum[idx - 1]

    return cum_sum

def quantizeImage(imgOrig:np.ndarray, nQuant:int, nIter:int)->(List[np.ndarray],List[float]):
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
    if isRGB(imgOrig):
        RGB=True
        imgYIQ = transformRGB2YIQ(imgOrig)
        Y=imgYIQ[:,:,0]
        unnormImg = unnormalize(Y).astype('int')
    else:
        RGB = False
        unnormImg = unnormalize(imgOrig).astype('int')

    img_lst=[imgOrig]
    err_lst=[]
    histOrig = calHist(unnormImg)
    h,w = unnormImg.shape[:2]

    partSize = (h* w) / nQuant
    z = [1]
    sum = 0
    for i in range(len(histOrig)):
        sum+=histOrig[i]
        if (sum>=partSize):
            z.append(i)
            sum=0

    z.append(255)

    for i in range(nIter):
        q = []
        for i in range(1,nQuant+1):
            cutHist=histOrig[z[i-1]:z[i]]
            avg=int(np.average(range(z[i-1], z[i]),axis=None, weights=cutHist, returned=False))
            q.append(avg)
        for i in range(1,nQuant):
            z[i]=int((q[i-1]+q[i])/2)

        img=np.zeros(unnormImg.shape)
        for i in range(0, nQuant):
            img[unnormImg>=z[i]]=q[i]
        errMat=pow((unnormImg-img),2)/(h*w)
        err=np.average(errMat)
        err_lst.append(err)


        if RGB:
            img = normalize(img)
            imgYIQ[:, :, 0] = img
            img = transformYIQ2RGB(imgYIQ)

        img_lst.append(img)


    return img_lst, err_lst
