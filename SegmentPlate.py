from __future__ import division
import cv2
import imutils
from imutils import paths, perspective
import numpy as np
import uuid
import os
from skimage.filters import threshold_local


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

            # c = max(cnts, key=cv2.contourArea)
            # rect = cv2.minAreaRect(c)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box*ratio)
            # cv2.drawContours(imgOrgi, [box], 0, (0, 255, 0), 2)

            # cv2.imshow('plate_rgb', plate_rgb)
            # cv2.imshow('imgOrgi', imgOrgi)
            # cv2.waitKey(0)
            return plate_rgb
    return None

# dirPath =  os.path.dirname(__file__)
# imgFolder = os.path.join(dirPath, 'imageBGR')
# pathsImg = paths.list_images(imgFolder)

# for path in pathsImg:
#     detect_plate(path)
#     k = cv2.waitKey(0)
#     if k == ord('q'):
#         exit()


cap = cv2.VideoCapture("20191110_3.h264")

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
i = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        plate_rgb = detect_plate(frame)
        if plate_rgb is not None:
            cv2.imshow('plate_rgb', plate_rgb)
            # Press Q on keyboard to  exit
            k = cv2.waitKey(1)
            if k == ord('d'):
                print('[INFO]: Dont save image')
                continue
            elif k == ord('q'):
                exit()
            else:
                name = uuid.uuid4()
                cv2.imwrite('dataShow/{}.jpg'.format(i), plate_rgb)
                i += 1
                print('[INFO]: Save image: {}'.format(name))
                exit()
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
