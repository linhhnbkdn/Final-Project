from __future__ import division
import cv2
import imutils
from imutils import paths, perspective
import numpy as np
import uuid
import os
import uuid

dirPath = os.path.dirname(os.path.realpath(__file__))


def segmenting_plate(plateBGR):
    plateBGROrg = plateBGR.copy()
    plate = imutils.resize(plateBGR.copy(), width=320)
    plateGray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    ratio = plateBGROrg.shape[0]/plate.shape[0]


    plateGray = cv2.GaussianBlur(plateGray, (3, 3), 0)
    edged = imutils.auto_canny(plateGray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 51))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    # eroded = cv2.erode(dilated, None, iterations=2)
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            cv2.rectangle(dilated, (startX, startY), (int(cx+WEIGHT/2), int(cy+HEIGHT/2)), 0, -1)

    WEIGHT = 110
    HEIGHT = 160
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea)

    if len(cnts) < 10:
        print('[INFO]: Image error')
        pass
    else:
        i = 0
        for c in cnts[-10:]:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cx = int(cx*ratio)
            cy = int(cy*ratio)
            # cv2.circle(plateBGROrg, (cx, cy), 10, (0, 255, 0), -1)
            startX = max(0, int(cx-WEIGHT/2))
            startY = max(0, int(cy-HEIGHT/2))
            endX = int(cx+WEIGHT/2)
            endY = int(cy+HEIGHT/2)
            numOfPlate = plateBGROrg[startY:endY, startX:endX]
            name = uuid.uuid4()
            cv2.rectangle(plateBGROrg, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.imwrite('dataShow/{}.jpg'.format(name), numOfPlate)
            print('[INFO]: Save image: {}'.format(name))

        # cv2.imshow('plateBGROrg', plateBGROrg)
        # cv2.waitKey(0)
            # cv2.rectangle(plateBGROrg, (startX, startY), (int(cx+WEIGHT/2), int(cy+HEIGHT/2)), (0, 255, 0))
    # plateBGROrg = imutils.resize(plateBGROrg, width=640)
    # cv2.imshow('plateBGROrg', plateBGROrg)
    # return cnts

# pathsImg = list(paths.list_images('{}/imgExportFromVideoBGR'.format(dirPath)))
pathsImg = list(paths.list_images('{}/dataShow'.format(dirPath)))
for i, pathImg in enumerate(pathsImg):
    plate = cv2.imread(pathImg)
    segmenting_plate(plate)

    k = cv2.waitKey(1)
    if k == ord('q'):
        exit()
    elif k == ord('d'):
        pass

