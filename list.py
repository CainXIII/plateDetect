import os,cv2, numpy as np
image_dir = 'plateData'
count = 0
for img in os.listdir(image_dir):
#    try:
        count += 1
        #print('Load: ', str(img))
        imgOriginal = cv2.imread(image_dir+'/'+img)
        height, width, numChannels = imgOriginal.shape
        #findChar(imgOriginal, count)
        imgHSV = np.zeros((height, width, 3), np.uint8)
        imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
        imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
        #print('Done: ', img)
        cv2.imwrite('PD/'+str(count)+'.jpg',imgHSV)
        print(imgValue)
#    except:
#        print('Error: ', img)
