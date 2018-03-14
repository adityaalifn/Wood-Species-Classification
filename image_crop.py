import os
import numpy as np
import random
from PIL import Image
rootdir = "dataset"
import cv2

def read_data() :
    image_list_train = []
    label_list_train = []
    image_list_test = []
    label_list_test = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.split(".")[-1] == "jpg":
                img = Image.open(os.path.join(subdir, file))
                img = img.convert("RGB")

                img_arr = np.asarray(img)
                for i in range(30):
                    start_y = random.randint(0,979) # Titik pengambilan pixel
                    start_x = random.randint(0, 416)
                    new_img_arr = img_arr[start_x:start_x+300, start_y:start_y+300, :]
                    #new_img_arr = new_img_arr.transpose((1,0,2))
                    #new_img_arr = np.expand_dims(new_img_arr,axis=0)
                    image_list_train.append(new_img_arr)
                    label_list_train.append(subdir.split("_")[-1]) # PENTING
                for i in range(10):
                    start_y = random.randint(0,979) # Titik pengambilan pixel
                    start_x = random.randint(716,723)
                    new_img_arr = img_arr[start_x:start_x+300, start_y:start_y+300, :]
                    #new_img_arr = new_img_arr.transpose((1,0,2))
                    #new_img_arr = np.expand_dims(new_img_arr,axis=0)
                    image_list_test.append(new_img_arr)
                    label_list_test.append(subdir.split("_")[-1]) # PENTING

    image_list_train = np.array(image_list_train)
    image_list_test = np.array(image_list_test)
    label_list_train = np.array(label_list_train)
    label_list_test = np.array(label_list_test)

    return image_list_train,label_list_train,image_list_test,label_list_test

def resize_imgs(imgs) :
    dim = (150, 150)
    resized = []
    for img in imgs :
        resized.append(cv2.resize(img, dim, interpolation=cv2.INTER_AREA))

    return np.array(resized)