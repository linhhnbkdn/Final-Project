import pickle
import os
import numpy
from sklearn.preprocessing import LabelBinarizer
import cv2

dir_path = os.path.dirname(__file__)

labelsPath = os.path.join(dir_path, 'MiniVGGNet', 'labelsSaved.h5')
labelsFile = open(labelsPath, 'rb')
labels = pickle.load(labelsFile)
labelsFile.close()

labels_class = numpy.argmax(labels, axis=1)
# # print(labels_class.shape)
# # print(labels_class[10000])
# unique, counts = numpy.unique(labels_class, return_counts=True)
# dataCount = dict(zip(unique, counts))
# print(dataCount)

dataPath = os.path.join(dir_path, 'MiniVGGNet', 'dataSaved.h5')
dataFile = open(dataPath, 'rb')
data = pickle.load(dataFile)
dataFile.close()

# data = data.astype("float") * 255

for index, image in enumerate(data):
    label = labels_class[index]
    cv2.imshow('image', image)
    image = image.astype("float") * 255
    cv2.imwrite('dataRecover/{}/{}.jpg'.format(label, index), image)
    print('dataRecover/{}/{}.jpg'.format(label, index))
    # cv2.waitKey(0)
    # exit()
# print(data.shape)
