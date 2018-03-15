from keras.optimizers import Adam,SGD
from keras.utils import to_categorical
from keras.models import Sequential
from keras import losses
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import Input
from keras.callbacks import ModelCheckpoint,EarlyStopping
import image_crop
import os
import cv2
from scipy.misc import imresize
from PIL import Image
import numpy as np

def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(150, 150, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(34, activation='softmax'))

    return model

if __name__ == '__main__':

    save_dir = os.path.join(os.getcwd(), 'saved_model')
    model_name = 'vgg16_wood.h5'
    weight_name = 'vgg16_wood.h5'

    x_train,y_train,x_test,y_test = image_crop.read_data()

    x_train_resized = image_crop.resize_imgs(x_train)

    y = to_categorical(y_train,num_classes=34)

    model = VGG_16()

    checkpoint = ModelCheckpoint('./saved_model/checkpoint.h5',verbose=1,save_best_only=True)
    earlystop = EarlyStopping(verbose=1)

    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt,loss=losses.categorical_crossentropy,metrics=[metrics.categorical_accuracy])
    model.fit(x_train_resized,y,epochs=10,batch_size=36,callbacks=[checkpoint,earlystop])

    model_path = os.path.join(save_dir,model_name)
    weight_path = os.path.join(save_dir,weight_name)
    model.save(model_path)
    model.save_weights(weight_path)
