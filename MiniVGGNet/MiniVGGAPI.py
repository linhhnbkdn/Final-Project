from sklearn.preprocessing import LabelBinarizer
from imageUtils.nn.conv import MiniVGGNet
from keras.optimizers import SGD
import numpy as np
import argparse
from imageUtils.datasets.simpledatasetloader import SimpleDatasetLoader
from imageUtils.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from imageUtils.preprocessing.simplepreprocessor import SimplePreprocessor
from glob import glob
import os
from sklearn.model_selection import train_test_split
import cv2

dirFolder = os.path.dirname(__file__)

im2Array = ImageToArrayPreprocessor('channels_last')
simplePre = SimplePreprocessor(55, 80)
loader = SimpleDatasetLoader([simplePre, im2Array])

# initialize the optimizer and model
print("[INFO] compiling model...")
model = MiniVGGNet.build(width=55, height=80, depth=3, classes=10)
modelPath = os.path.join(dirFolder, 'weightPathCPU.h5')
model.load_weights(modelPath)

def miniVGGAPI(load_image_array):
    data = loader.load_image_array(load_image_array)
    data = data.astype("float") / 255.0
    yHat = model.predict(data)
    y = np.argmax(yHat, axis=1)
    yHat = yHat.reshape(-1, 1)
    yProba = yHat[y[0]]
    return (y[0], yProba[0]*100)




