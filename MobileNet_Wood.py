from keras.optimizers import RMSprop
from keras.applications import MobileNet
from keras.utils import to_categorical
from keras import losses, metrics
from keras.callbacks import ModelCheckpoint
import image_crop
from keras.preprocessing.image import ImageDataGenerator
import os

if __name__ == '__main__':
    train_datagen = ImageDataGenerator(
        vertical_flip=True,
        rotation_range=30
    )

    test_datagen = ImageDataGenerator(
        vertical_flip=True,
        rotation_range=30
    )

    train_generator = train_datagen.flow_from_directory(
        'Dataset_8class/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
        # save_to_dir="./augmented"
    )

    validation_generator = test_datagen.flow_from_directory(
        'Dataset_8class/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'MobileNet_wood_model_new.hdf5'
    weight_name = 'MobileNet_wood_weight_new.hdf5'
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)

    # x_train,y_train,x_test,y_test = image_crop.read_data()

    # x_train_resized = image_crop.resize_imgs(x_train)

    # y = to_categorical(y_train,num_classes=34)

    model = MobileNet(include_top=True, weights=None, classes=8,
                      pooling='max', input_shape=(150, 150, 3))
    checkpoint = ModelCheckpoint(filepath=os.path.join(
        save_dir, 'MobileNet_weight.{epoch:02d}-{loss:.2f}.hdf5'), verbose=1)

    opt = RMSprop(lr=2e-5)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=[
                  metrics.categorical_accuracy])
    # model.fit(x_train_resized,y,epochs=10,batch_size=36,callbacks=[checkpoint])
    model.fit_generator(
        train_generator,
        steps_per_epoch=300,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=100
    )

    model.save(model_path)
    model.save_weights(weight_path)
