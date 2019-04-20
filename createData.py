import cv2, numpy as np
from PIL import Image
import math,os

#imgOriginal = cv2.imread('image/86468.jpg')


#height, width, numChannels = imgOriginal.shape
# module
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
#preprocess
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh
#extractValue
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue
# maximizeContrast
def maximizeContrast(imgGrayscale):
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat

def findplate(imgOriginal):
    imgGrayscaleScene, imgThreshScene = preprocess(imgOriginal)
    height, width, numChannels = imgOriginal.shape
    plate = np.zeros((height,width,1),np.uint8)
    _, contours,hierarchy = cv2.findContours(imgThreshScene, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
        if (area >= 8000)and((len(approx)==8)or(len(approx)==4)):
            #crop
            x,y,w,h = cv2.boundingRect(contour)
            plate = imgOriginal[y:y+h,x:x+w]
            #cv2.imshow('img',new_img)
    return plate

#import class Char
import Char
def checkIfPossibleChar(Char):
    if (Char.intBoundingRectArea > 30 and
        Char.intBoundingRectWidth > 5 and Char.intBoundingRectHeight > 10 and
        0.15 < Char.fltAspectRatio and Char.fltAspectRatio < 1.0):
        return True
    else:
        return False
path = ''
def findChar(imgOriginal, path):

    new_img = findplate(imgOriginal)
    imgGrayscale, imgThresh = preprocess(new_img)
    adaptivePlate = cv2.adaptiveThreshold(imgGrayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    blurPlate = cv2.GaussianBlur(adaptivePlate, (5,5),0)
    ret, processedPlate = cv2.threshold(blurPlate,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    listOfPossibleChars = []
    ind = 0
    area_char = []
    _,contours,_ = cv2.findContours(processedPlate,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        char = Char.Char(contour)
        if (checkIfPossibleChar(Char.Char(contour))):
            listOfPossibleChars.append(contour)
            area_char.append(cv2.contourArea(contour))
    #listOfPossibleChars.sort(key = lambda Char: Char.intCenterX)
    avg = sum(area_char)/len(area_char)
    for char in listOfPossibleChars:
        if (cv2.contourArea(char)>avg):
            x,y,w,h = cv2.boundingRect(char)
            img_char = new_img[y:y+h,x:x+w]
            ind += 1
            #cv2.imshow(str(ind) ,img_char)
            cv2.imwrite('data/'+str(path)+str(ind)+'.jpg',img_char)

image_dir = 'plateData'
count = 0
for img in os.listdir(image_dir):
    try:
        count += 1
        print('Load: ', str(img))
        imgOriginal = cv2.imread(image_dir+'/'+img)
        findChar(imgOriginal, count)
        print('Done: ', img)
    except:
        print('Error: ', img)

print("Created data!")
#findChar(imgOriginal, path)
#cv2.drawContours(new_img, listOfPossibleChars, -1, (0,255,0),3)
#cv2.imshow('img',imgOriginal)
#cv2.waitKey(0)
cv2.destroyAllWindows()
