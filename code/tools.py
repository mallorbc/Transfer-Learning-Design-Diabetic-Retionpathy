import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout,Input,concatenate
import sys
# sys.path.insert(1, '../diabetic-detection')
# sys.path.append("../diabetic-detection/code")
import argparse
import os
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from myModels import *
from utils import *
from plots import *
import copy
from random import randint



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Activation map tester')
    parser.add_argument("-model",default=None,help="Path to the model",type=str)
    parser.add_argument("-img","--image",default=None,help="Image to test on",type=str)
    parser.add_argument("-csv","--csv_location",default=None,help="location of the csv data file",type=str)
    parser.add_argument("-test_csv","--test_csv_location",default=None,help="The file the can be used to further reduce files created",type=str)
    parser.add_argument("-class","--class_to_test",default=None,help="What class we are going to be testing on",type=int)
    parser.add_argument("-m","--mode",default=None,help="What mode to run for metrics",type=int)
    parser.add_argument("-o","--output",default=None,help="Where to save outputs",type=str)
    parser.add_argument("-npy",default=None,help="npy file to view",type=str)


    args = parser.parse_args()
    image_to_test = args.image
    csv_dir = args.csv_location
    csv_dir = os.path.realpath(csv_dir)
    test_csv = args.test_csv_location
    if test_csv is not None:
        test_csv = os.path.realpath(test_csv)
    class_to_test = args.class_to_test
    mode_to_run = args.mode
    output_folder = args.output
    npy_file = args.npy

    health_level,image_name = load_data(csv_dir)
    


    
    if os.path.isfile(image_to_test):
        print("testing one image")
        image_name = os.path.abspath(args.image)        
        health_dict = make_csv_dict(health_level,image_name)
        image_to_test = [image_name]
    else:
        print("Will test on all images in csv")
        data_path = os.path.abspath(args.image)
        image_name = get_full_image_name_no_ext(data_path,image_name)
        health_dict = make_csv_dict(health_level,image_name)
        image_to_test = image_name




    if mode_to_run == 1:
        #loads the model
        model_to_load = os.path.realpath(args.model)
        model = load_model(model_to_load)
        make_precision_recall_curve(class_to_test,image_to_test,health_dict,model)
    
    elif mode_to_run == 2:
        #loads the model
        model_to_load = os.path.realpath(args.model)
        model = load_model(model_to_load)
        make_roc_curve(class_to_test,image_to_test,health_dict,model)

    elif mode_to_run == 3:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        make_npy_of_class(class_to_test,image_to_test,health_dict,output_folder,test_csv)
    
    elif mode_to_run ==4:
        npy_file = os.path.realpath(npy_file)
        image = np.load(npy_file)
        print(np.shape(image))
        for i,m in enumerate(image):
            plt.imshow(m)
            plt.show()

    elif mode_to_run == 5:
        base_path = args.image
        images = os.listdir(args.image)
        number_of_images = len(images)

        # image_to_test = images[0]
        image_to_test = images[randint(0,(number_of_images-1))]
        image_to_test = np.load(base_path+"/"+image_to_test)

        print(np.shape(image_to_test))
        # image_to_test = np.squeeze(image_to_test,axis=0)
        # image_to_test = cv2.imread(image_to_test)
        # image_to_test = image_to_test * 255.0
        # quit()
        image = Image.fromarray(image_to_test,mode="RGB")
        image.show()

        
    elif mode_to_run == 6:
        if class_to_test is not None:
            image_name = get_images_of_one_class(class_to_test,image_name,health_dict)

        image_name = add_extension(image_name,".jpeg")

        number_of_images = len(image_name)
        image_to_test = image_name[randint(0,(number_of_images-1))]
        image_to_test = cv2.imread(image_to_test)

        image_to_test = cv2.cvtColor(image_to_test, cv2.COLOR_BGR2RGB)
        plt.imshow(image_to_test)
        plt.show()




    print(mode_to_run)





    # model_weights = model.get_weights()
    # print(model_weights)
    # new_image_width = 512
    # new_image_height = 512
    # new_model = transfer_learning_model_inception_v3(new_image_width, new_image_height,True)
    # # new_model.set_weights(model_weights)
    # for layer in new_model.layers:
    #     print(layer)
    # # print(new_model.get_layer("convd2_93"))
    # quit()
    # image = np.squeeze(image,axis=-1)
    # print(np.shape(image))
    # quit()

    # print(model.predict_classes(image))
    # new_image_width = 512
    # new_image_height = 512
    # base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    # base_model.summary()
    # print(base_model.get_layer("conv2d_93"))
    # model2 = models.load_model(model.get_layer(index=0))

    # for layer in model.layers:
    #     print(layer)


