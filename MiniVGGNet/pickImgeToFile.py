import pickle
from imageUtils.datasets.simpledatasetloader import SimpleDatasetLoader
from imageUtils.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from imageUtils.preprocessing.simplepreprocessor import SimplePreprocessor
import os
from glob import glob
from sklearn.preprocessing import LabelBinarizer

im2Array = ImageToArrayPreprocessor('channels_last')
simplePre = SimplePreprocessor(55, 80)

dirFolder = os.path.dirname(__file__)
imFolder = os.path.join(dirFolder, '..','data')
pathsImg = glob('{}/*/*.jpg'.format(imFolder))

dataFileSaved = os.path.join(dirFolder, 'dataSaved.h5')
labelsFileSaved = os.path.join(dirFolder, 'labelsSaved.h5')
loader = SimpleDatasetLoader([simplePre, im2Array])
data, labels = loader.load(pathsImg, verbose=10)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
data = data.astype("float") / 255.0


with open(dataFileSaved, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(labelsFileSaved, 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


