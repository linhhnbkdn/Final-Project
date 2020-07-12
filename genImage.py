# from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
# from matplotlib import pyplot as plt
import cv2
import imutils
import os

IMG_WIDTH = 55
IMG_HEIGHT = 80
batch_size = 1
NUM_GEN = 5000
image_generator = ImageDataGenerator(zoom_range=[1, 1.5],
                                    rotation_range=10,
                                    height_shift_range=0.15,
                                    width_shift_range=0.15,
                                    brightness_range=[0.2, 2],
                                    shear_range=0.15)

for index in range(1, 2, 1):
    # os.system('rm -rf dataRecover/*')
    # os.system('cp -r _dataRecover/{} dataRecover/'.format(index))
    print('done copy')
    gen = image_generator.flow_from_directory(directory='dataRecover/',
                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                            batch_size=1, class_mode=None,
                                            save_format='jpg', classes=None,
                                            color_mode='rgb', save_to_dir='data/{}'.format(index),
                                            save_prefix='{}'.format(index))

    i = 0
    for batch in gen:
        i += 1
        print(i, index)
        if i > NUM_GEN:
            break

