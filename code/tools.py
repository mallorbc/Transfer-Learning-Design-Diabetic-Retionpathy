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
from keract import get_activations, display_heatmaps




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
    parser.add_argument("-model_num",default=1,help="What model type to load",type=int)
    parser.add_argument("-compat","--compatibility_mode",default=False,type=str2bool)
    parser.add_argument("-width",default=512,help="What is the width of the image",type=int)
    parser.add_argument("-height",default=512,help="What is the width of the image",type=int)
    parser.add_argument("-n","--name",default=None,help="names for saves",type=str)
    parser.add_argument("-trainable",default=None,help="freeze the weights or not",type=str2bool)



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
    model_num = args.model_num
    compat_mode = args.compatibility_mode
    width = args.width
    height = args.height
    name = args.name
    unfrozen_weights = args.trainable

    if output_folder is not None:
        output_folder = os.path.realpath(output_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    if compat_mode is True:
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

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
        image_name = add_extension(image_name,".jpeg")
        health_dict = make_csv_dict(health_level,image_name)
        image_to_test = image_name



    #makes precision recall curve
    if mode_to_run == 1:
        #loads the model
        model_to_load = os.path.realpath(args.model)
        model = load_model(model_to_load,model_num,width,height)
        make_precision_recall_curve(class_to_test,image_to_test,health_dict,model)
    
    #makes ROC curve
    elif mode_to_run == 2:
        #loads the model
        model_to_load = os.path.realpath(args.model)
        model = load_model(model_to_load,model_num,width,height)
        make_roc_curve(class_to_test,image_to_test,health_dict,model)

    #makes numpy files of images for GAN
    elif mode_to_run == 3:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        make_npy_of_class(class_to_test,image_to_test,health_dict,output_folder,test_csv)
    
    #for loading a numpy file and viewing it
    elif mode_to_run ==4:
        npy_file = os.path.realpath(npy_file)
        image = np.load(npy_file)
        print(np.shape(image))
        for i,m in enumerate(image):
            plt.imshow(m)
            plt.show()

    #view the image in PIL
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

    #views one of the images in matplotlib
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
    elif mode_to_run == 7:
        # transfer_learning_model_inception_v3_functional(512, 512)
        # quit()
        #loads the full path
        # print(class_to_test)
        if class_to_test is not None:
            images_of_one_class = get_images_of_one_class(class_to_test,image_to_test,health_dict)
            # images_of_one_class = add_extension(images_of_one_class,".jpeg")
            image_to_test = images_of_one_class[randint(0,(len(images_of_one_class)-1))]
        else:
            image_to_test = image_to_test[0]



        model_to_load = os.path.realpath(args.model)
        #loads the model with the weights
        model = load_model(model_to_load,model_num,width,height)
        #converts array to string
        
        #reads the image
        img_data = cv2.imread(image_to_test)
        print("image read successful")
        #converts the data into a np array
        img_arr = np.asarray(img_data)
        print(np.shape(img_arr))
        #reshapes the data 
        img_arr.reshape(width, height,3)
        #expands the dimensions, needed for api
        img_arr = np.expand_dims(img_arr, axis=0)
        print(np.shape(img_arr))
        # quit()
        #gets al activations for conv layers
        img_activation = get_activations(model, img_arr, auto_compile=True)

        #uncomment this to see keys
        # print(img_activation.keys())
        # quit()

        #key used for simple cnn, may need changed
        layer_to_use = img_activation["conv2d_14"]
        #last layer for functional
        # layer_to_use = img_activation["conv2d_93"]


        #removes first dimension that was needed to get activations
        layer_to_use = np.squeeze(layer_to_use,axis=0)
        #gets the average of all the filters so we can have one image instead of 100s
        layer_to_use = np.mean(layer_to_use,axis=-1)
        #increases resolution of the activation
        activation_resized = cv2.resize(layer_to_use,(width,height))
        # activation_resized_added_channel = np.expand_dims(activation_resized,axis=-1)
        # activation_resized_added_channel = np.tile(activation_resized_added_channel,(1,1,3))
        print(np.shape(activation_resized))
        print(activation_resized.dtype)

        #removes chanel dimension so we can overlay data
        img_data = img_data[:, :, 0]

        #converts image data to float32
        img_data = img_data.astype("float32")

        #overlays the data, may need to tweak parameters
        overlay = cv2.addWeighted(src1=activation_resized, alpha=1, src2=img_data, beta=1, gamma=0)
        #shows just the activation
        plt.imshow(activation_resized)
        plt.show()
        #displays overlay
        plt.imshow(overlay)
        plt.show()





        #dictionary that will be passed back to api
        layer_act = {}
        #puts the layer we want in the dictionary
        layer_act[1] = layer_to_use
        # display_heatmaps(layer_act, img_arr, save=False)

    elif mode_to_run == 8:
        if test_csv is not None:
            test_labels,test_images = load_data(test_csv)
            #gets the full paths of the images
            test_images = get_full_image_name_no_ext(data_path,test_images)

            # test_images = get_full_image_name_no_ext(data_path,test_images)
            image_to_test = test_images

        model_to_load = os.path.realpath(args.model)
        model = load_model(model_to_load,model_num,width,height,unfrozen_weights)
        # image_to_test = add_extension(image_to_test,".jpeg")
        output_file = output_folder + "/confusion_matrix.png"
        create_confusion_matrix_one_input(model,image_to_test,test_labels,name,output_file)
        make_roc_precision_recall_graphs(0,image_to_test,health_dict,model,output_folder)
        make_roc_precision_recall_graphs(1,image_to_test,health_dict,model,output_folder)
        make_roc_precision_recall_graphs(2,image_to_test,health_dict,model,output_folder)
        make_roc_precision_recall_graphs(3,image_to_test,health_dict,model,output_folder)
        make_roc_precision_recall_graphs(4,image_to_test,health_dict,model,output_folder)


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


