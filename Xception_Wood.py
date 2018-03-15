from keras.optimizers import RMSprop
from keras.applications import Xception
from keras.utils import to_categorical
from keras import losses,metrics
from keras.callbacks import ModelCheckpoint
import image_crop
import os

if __name__ == '__main__':

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'xception_wood_model.hdf5'
    weight_name = 'xception_wood_weight.hdf5'
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)

    x_train,y_train,x_test,y_test = image_crop.read_data()

    x_train_resized = image_crop.resize_imgs(x_train)

    y = to_categorical(y_train,num_classes=34)

    model = Xception(include_top=True,weights=None,classes=34)

    checkpoint = ModelCheckpoint(filepath=os.path.join(save_dir,'weight.{epoch:02d}-{loss:.2f}.hdf5'),verbose=1)

    opt = RMSprop(lr=2e-5)
    model.compile(optimizer=opt,loss=losses.categorical_crossentropy,metrics=[metrics.categorical_accuracy])
    model.fit(x_train_resized,y,epochs=10,batch_size=36,callbacks=[checkpoint])

    model.save(model_path)
    model.save_weights(weight_path)