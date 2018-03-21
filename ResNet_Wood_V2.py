from keras.optimizers import RMSprop
from keras.applications import InceptionResNetV2
from keras.utils import to_categorical
from keras import losses,metrics
from keras.callbacks import ModelCheckpoint
import Dataset_mobile
import os

if __name__ == '__main__':

    save_dir = os.path.join(os.getcwd(), 'saved_model')
    model_name = 'ResNetV2_wood_model.hdf5'
    weight_name = 'ResNetV2_wood_weight.hdf5'
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)

    x_train,y_train,x_test,y_test,dictionary = Dataset_mobile.read_data()

    x_train_resized = Dataset_mobile.resize_imgs(x_train,150)
    x_test_resized = Dataset_mobile.resize_imgs(x_test,150)

    y_train = to_categorical(y_train,num_classes=2)
    y_test = to_categorical(y_test,num_classes=2)

    model = InceptionResNetV2(include_top=True,weights=None,classes=2)

    model.load_weights(weight_path)

    # checkpoint = ModelCheckpoint(filepath=os.path.join(save_dir,'ResNetV2_weight.{epoch:02d}-{loss:.2f}-{categorical_accuracy:.2f}.hdf5'),verbose=1,monitor='categorical_accuracy',save_best_only=True)

    opt = RMSprop(lr=2e-5)
    model.compile(optimizer=opt,loss=losses.categorical_crossentropy,metrics=[metrics.categorical_accuracy])
    # model.fit(x_train_resized,y_train,epochs=20,batch_size=6)
    #
    # model.save_weights(weight_path)
    # model.save(model_path)

    score1 = model.evaluate(x_train_resized, y_train, batch_size=6)
    score2 = model.evaluate(x_test_resized, y_test, batch_size=6)
    print(score1)
    print(score2)