from keras.optimizers import Adam,RMSprop
from keras.applications import MobileNet
from keras.utils import to_categorical
from keras import losses,metrics
from keras.callbacks import ModelCheckpoint
import image_crop
import os
from keras import backend as K
import Dataset_mobile
import numpy as np

if __name__ == '__main__':

    # K.tensorflow_backend._get_available_gpus()
    save_dir = os.path.join(os.getcwd(), 'saved_model')
    model_name = 'MobileNet_wood_model.hdf5'
    weight_name = '1_MobileNet_wood_weight.hdf5'
    saved_weight = '3_MobileNetV2_weight.24-0.21-1.00.hdf5'
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)
    saved_weight_path = os.path.join(save_dir, saved_weight)

    x_train,y_train,x_test,y_test,dictionary = Dataset_mobile.read_data()

    x_train_resized = image_crop.resize_imgs(x_train)
    x_test_resized = image_crop.resize_imgs(x_test)

    y_train = to_categorical(y_train,num_classes=2)
    y_test = to_categorical(y_test,num_classes=2)

    print(np.shape(x_train_resized))
    print(np.shape(x_test_resized))
    model = MobileNet(include_top=True,weights=None,classes=2,pooling='max',input_shape=(200,200,3))

    model.load_weights(weight_path)

    checkpoint = ModelCheckpoint(filepath=os.path.join(save_dir,'MobileNetV2_weight.{epoch:02d}-{loss:.2f}-{categorical_accuracy:.2f}.hdf5'),verbose=1,monitor='categorical_accuracy',save_best_only=True)

    opt = Adam(lr=5e-6)
    model.compile(optimizer=opt,loss=losses.categorical_crossentropy,metrics=[metrics.categorical_accuracy])
    # model.fit(x_train_resized,y_train,epochs=20,batch_size=6,callbacks=[checkpoint])
    #
    # model.save(model_path)
    # model.save_weights(weight_path)

    score1 = model.evaluate(x_train_resized,y_train,batch_size=6)
    score2 = model.evaluate(x_test_resized,y_test,batch_size=6)
    print(score1)
    print(score2)