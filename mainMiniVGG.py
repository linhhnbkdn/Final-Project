from __future__ import division
import cv2
import imutils
from imutils import paths, perspective
from imutils.contours import sort_contours
import numpy as np
import uuid
import os
import uuid
from MiniVGGNet.MiniVGGAPI import miniVGGAPI
from skimage.filters import threshold_local

dirPath = os.path.dirname(os.path.realpath(__file__))


def detect_plate(pathImg):
    # imgOrgi = cv2.imread(pathImg)
    imgOrgi = pathImg
    img = imutils.resize(imgOrgi, width=360, inter=cv2.INTER_CUBIC)
    ratio = imgOrgi.shape[0]/img.shape[0]
    print('[INFO]: Ratio {}'.format(ratio))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = cv2.split(cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)

    dilated = cv2.dilate(thresh, None, iterations=2)
    erode = cv2.erode(dilated, None, iterations=2)

    # cv2.imshow('erode', erode)

    cnts = cv2.findContours(
        erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    offsetX = 100
    offsetY = 30
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        print('[INFO]: Image Rezied shape : {} {}'.format(height, width))
        if height == 0:
            return None
        cond_div = width/height > 4 and width/height < 5
        cond_height = (height > 65) and (height < 70)
        if cond_div and cond_height:
            box = box*ratio
            plate_rgb = perspective.four_point_transform(imgOrgi, box)
            plate_rgb = plate_rgb[offsetY:plate_rgb.shape[0] -
                                  offsetY, offsetX: plate_rgb.shape[1] - offsetX]
            print('[INFO]: We have image')
            return plate_rgb
    return None


def segmenting_plate(plateBGR):
    plateBGROrg = plateBGR.copy()
    plate = imutils.resize(plateBGR.copy(), width=320)
    plateGray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    ratio = plateBGROrg.shape[0]/plate.shape[0]
    plateGray = cv2.GaussianBlur(plateGray, (3, 3), 0)
    edged = imutils.auto_canny(plateGray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 51))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    cnts = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea)
    WEIGHT = 50
    HEIGHT = 160
    for c in cnts:
        S = cv2.contourArea(c)
        if S > 2000:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            startX = max(0, int(cx-WEIGHT/2))
            startY = max(0, int(cy-HEIGHT/2))
            cv2.rectangle(dilated, (startX, startY),
                          (int(cx+WEIGHT/2), int(cy+HEIGHT/2)), 0, -1)

    WEIGHT = 110
    HEIGHT = 160
    cnts = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea)
    numImg = list()

    if len(cnts) < 10:
        print('[INFO]: Image error')
        pass
    else:
        cnts = cnts[-10:]
        cnts, boundingBoxes = sort_contours(cnts, method='left-to-right')
        for c in cnts:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cx = int(cx*ratio)
            cy = int(cy*ratio)
            startX = max(0, int(cx-WEIGHT/2))
            startY = max(0, int(cy-HEIGHT/2))
            endX = int(cx+WEIGHT/2)
            endY = int(cy+HEIGHT/2)
            numOfPlate = plateBGROrg[startY:endY, startX:endX]
            numImg.append(numOfPlate)
    return numImg, plateBGROrg


# pathsImg = list(paths.list_images('{}/imageBGR'.format(dirPath)))
# for i, pathImg in enumerate(pathsImg):
#     img = cv2.imread(pathImg)
#     cv2.imshow('Input', img)
#     plate = detect_plate(img)
#     if plate is not None:
#         numImg, plateBGROrg = segmenting_plate(plate)
    # for num in numImg:
    #     (y, yProba) = miniVGGAPI(num)
    #     num = cv2.putText(num, '{}'.format(y), (5, 20), cv2.FONT_HERSHEY_COMPLEX,
    #                1, (255, 0, 0), 1)
    #     num = cv2.putText(num, '{}%'.format(yProba), (5, 60), cv2.FONT_HERSHEY_COMPLEX,
    #                1, (255, 0, 0), 1)
    #     cv2.imshow('num', num)

    #     print(y, yProba)
    #     cv2.waitKey(0)

cap = cv2.VideoCapture("20191110_3.h264")

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        plate = detect_plate(frame)
        if plate is not None:
            numImg, plateBGROrg = segmenting_plate(plate)
            i = 5
            for num in numImg:
                (y, yProba) = miniVGGAPI(num)
                frame = cv2.putText(frame, '{} '.format(y), (i, 100), cv2.FONT_HERSHEY_COMPLEX,
                                        3, (255, 0, 0), 5)
                # frame = cv2.putText(frame, '{}%'.format(yProba), (i, 60), cv2.FONT_HERSHEY_COMPLEX,
                #                         1, (255, 0, 0), 1)
                i = i + 50
                # cv2.imshow('numOfPlate', numOfPlate)
                # cv2.imshow('Plate', a)
        frame = imutils.resize(frame, width=720)
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
