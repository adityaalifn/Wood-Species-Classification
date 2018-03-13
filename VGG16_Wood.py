from keras.applications.vgg16 import VGG16
from keras.optimizers import Nadam


model = VGG16(include_top=False,weights='none',input_shape=(100,100,3),pooling='max')

opt = Nadam()
model.compile(optimizer=opt,loss='binary_crossentropy')