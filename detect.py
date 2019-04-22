import cv2, numpy as np
from PIL import Image
import math,sys

# module
from keras.models import model_from_json
from keras.models import load_model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

label_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
#preprocess
def preprocess(image):
    imgGrayscale = extractValue(image)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh
#extractValue
def extractValue(image):
    height, width, numChannels = image.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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

def findplate(image):
    imgGrayscaleScene, imgThreshScene = preprocess(image)
    height, width, numChannels = image.shape
    plate = np.zeros((height,width,1),np.uint8)
    _, contours,hierarchy = cv2.findContours(imgThreshScene, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
        if (area >= 8000)and((len(approx)==4)):
        #if len(approx)==4:
            #crop
            x,y,w,h = cv2.boundingRect(contour)
            plate = image[y:y+h,x:x+w]
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
def findChar(image):
    new_img = findplate(image)
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
            img_char = cv2.resize(img_char,(20,20))
            img_char = np.expand_dims(img_char, axis = 0)
            #print(label_char[int(np.max(loaded_model.predict(img_char)))])
            print(label_char[np.argmax(loaded_model.predict(img_char))])
            #cv2.imshow(str(ind) ,img_char)

path = sys.argv[1]
imgOriginal = cv2.imread('plateData/'+path)
findChar(imgOriginal)
#cv2.drawContours(new_img, listOfPossibleChars, -1, (0,255,0),3)
#cv2.imshow('img',imgOriginal)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
