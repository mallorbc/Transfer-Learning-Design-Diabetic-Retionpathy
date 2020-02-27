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

    # print(health_level, image_name)

    
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
    print(health_dict)



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

        compat_mode = True
        if compat_mode is True:
            from tensorflow.compat.v1 import ConfigProto
            from tensorflow.compat.v1 import InteractiveSession

            config = ConfigProto()
            config.gpu_options.allow_growth = True
            session = InteractiveSession(config=config)
        
        width =512
        height =512
        
        #loads the full path
        if args.model is not None:
            model_to_load = os.path.realpath(args.model)

        #loads the model with the weights
        if args.model is not None:
            model = load_model(model_to_load,model_num,width,height)
        else:
            if model_num == 1:
                model = create_CNN(width,height)
            elif model_num == 2:
                model = transfer_learning_model_inception_v3_functional(width,height)

        #get image of one class, image_of_class uses full path
        image_of_class = get_images_of_one_class(class_to_test, image_name, health_dict)
        # print(image_of_class)
        # quit()
        image_of_class_normalized = normalize_images(image_of_class)
        #predicting classes from one class image
        predicted_class_level = custom_predict_class(model,image_of_class_normalized)

        #generate correct corrresponding class labels
        true_class_level = []
        for i in range(len(image_of_class)):
            # print(image_name[i])
            data_path = os.path.basename(image_of_class[i])
            # print(data_path)
            
            correct_label = health_dict[data_path]
            # print(correct_label)
            
            true_class_level.append(correct_label)      

        print("true class level and predicted class level obtained !!")

        print("start generating list")
        #create correct-prediction list and incorrect-prediction list
        correct_list,incorrect_list, incorrect_label= create_listcreate_list(predicted_class_level, true_class_level,image_of_class)
        # incorrect_label.astype(str)

        #derive correct image list and incorrect image list
        correct_image_list = []
        incorrect_image_list = []
        for i in correct_list:
            correct_image_name = image_of_class[i]
            correct_image_list.append(correct_image_name)
        for i in incorrect_list:
            incorrect_image_name = image_of_class[i]
            incorrect_image_list.append(incorrect_image_name)
        print("number of class image")
        print(len(image_of_class))
        print("2 lists generated Successfully !")
        print("number of correctly predicted images")
        print(len(correct_image_list))
        # print(correct_image_list)
        print("number of incorrectly predicted images")
        print(len(incorrect_image_list))
        # print(incorrect_image_list)
        
        # quit()    
        #converts array to string

        image_to_test = image_to_test[0]
        
        correct_list_basename = []
        incorrect_list_basename = []
        # find correct_image_list basename
        for i in range(len(correct_image_list)):
            data_path = os.path.basename(correct_image_list[i])
            correct_list_basename.append(data_path)
        print(correct_list_basename)
        for i in range(len(incorrect_image_list)):
            data_path = os.path.basename(incorrect_image_list[i])
            incorrect_list_basename.append(data_path)
        print(incorrect_list_basename)
        
        # Visulization_plots_correct = "/home/zhang4ym/Desktop/Senior-Design-Diabetic-Retionpathy/code/Visulization_plots/correctly predicted/"
        # Visulization_plots_incorrect = "/home/zhang4ym/Desktop/Senior-Design-Diabetic-Retionpathy/code/Visulization_plots/incorrectly predicted/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        class_dir = '/Class ' + str(class_to_test)
        correct_dir = output_folder + class_dir + '/correct/'
        incorrect_dir = output_folder + class_dir + '/incorrect/'
        # os.path.join(output_folder, "/correct/")
        # os.path.join(output_folder, "/incorrect")
        
        if not os.path.exists(correct_dir):
            os.makedirs(correct_dir)
        
        if not os.path.exists(incorrect_dir):
            os.makedirs(incorrect_dir)
        
        count_correct_list = 0
        count_incorrect_list = 0
        layer_name = "conv2d_14"
        for image in correct_image_list:
            #reads the image
            image = cv2.imread(image)
            print("image read successful")

            #converts the data into a np array
            image= np.asarray(image)
            #reshapes the data 
            image = image.reshape(width, height,3)
            image_filename = correct_list_basename[count_correct_list]+ '-original.png'
            # plt.imshow(image)
            # plt.show()
            plt.figure()
            plt.imshow(image)
            plt.savefig(os.path.join(correct_dir, image_filename))
            #expands the dimensions, needed for api
            image = np.expand_dims(image, axis=0)
            #gets al activations for conv layers
            img_activation = get_activations(model, image, auto_compile=True)
        
            #uncomment this to see keys
            # print(img_activation.keys())
        
            #key used for simple cnn, may need changed
            layer_to_use = img_activation[layer_name]
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
            image = image[0,:, :, 0]
            print(np.shape(image))
            print(image.dtype)
            
            
            #converts image data to float32
            image = image.astype("float32")
            #shows just the activation
            # plt.imshow(activation_resized)
            # plt.show()
            # quit()
            
            # find basename of path for image name to save
            # data_path = os.path.basename([image])
            #overlays the data, may need to tweak parameters
            # overlay = cv2.addWeighted(src1=activation_resized, alpha=1, src2=image, beta=1, gamma=0)
            overlay = cv2.addWeighted(src1=activation_resized, alpha=1, src2=image, beta=0.2, gamma=0)

            # #displays overlay
            overlay_filename = correct_list_basename[count_correct_list]+ '-overlay.png'
            average_activation_filename = correct_list_basename[count_correct_list]+ '-average_activation.png'
            plt.figure()
            plt.imshow(overlay)
            plt.savefig(os.path.join(correct_dir, overlay_filename))
            plt.figure()
            plt.imshow(activation_resized)
            plt.savefig(os.path.join(correct_dir, average_activation_filename))
            


            print("saving correct image")
            print(correct_list_basename[count_correct_list])
            #shows just the activation
            # plt.imshow(activation_resized)
            # plt.show()
            
            
            count_correct_list = count_correct_list + 1

        for image in incorrect_image_list:
            #reads the image
            image = cv2.imread(image)
            print("image read successful")

            #converts the data into a np array
            image= np.asarray(image)
            #reshapes the data 
            image = image.reshape(width, height,3)
            class_to_test = str(class_to_test)
            image_filename = incorrect_list_basename[count_incorrect_list]+ " from " + class_to_test + " to "+ incorrect_label[count_incorrect_list]+'-original.png'
            # plt.imshow(image)
            # plt.show()
            plt.figure()
            plt.imshow(image)
            plt.savefig(os.path.join(incorrect_dir, image_filename))
            #expands the dimensions, needed for api
            image = np.expand_dims(image, axis=0)
            #gets al activations for conv layers
            img_activation = get_activations(model, image, auto_compile=True)
        
            #uncomment this to see keys
            # print(img_activation.keys())
        
            #key used for simple cnn, may need changed
            layer_to_use = img_activation[layer_name]
            #removes first dimension that was needed to get activations
            layer_to_use = np.squeeze(layer_to_use,axis=0)
            #gets the average of all the filters so we can have one image instead of 100s
            layer_to_use = np.mean(layer_to_use,axis=-1)
            #increases resolution of the activation
            activation_resized = cv2.resize(layer_to_use,(width,height))
            # activation_resized_added_channel = np.expand_dims(activation_resized,axis=-1)
            # activation_resized_added_channel = np.tile(activation_resized_added_channel,(1,1,3))
            # print(np.shape(activation_resized))
            # print(activation_resized.dtype)
            
           
            #removes chanel dimension so we can overlay data
            image = image[0,:, :, 0]
            print(np.shape(image))
            print(image.dtype)
            
            
            #converts image data to float32
            image = image.astype("float32")
            #shows just the activation
            # plt.imshow(activation_resized)
            # plt.show()
            # quit()
            
            # find basename of path for image name to save
            # data_path = os.path.basename([image])
            #overlays the data, may need to tweak parameters
            # overlay = cv2.addWeighted(src1=activation_resized, alpha=1, src2=image, beta=1, gamma=0)
            overlay = cv2.addWeighted(src1=activation_resized, alpha=1, src2=image, beta=0.2, gamma=0)

            # #displays overlay
            overlay_filename = incorrect_list_basename[count_incorrect_list]+ " from " + class_to_test + " to "+ incorrect_label[count_incorrect_list] + '-overlay.png'
            average_activation_filename = incorrect_list_basename[count_incorrect_list]+ " from " + class_to_test + " to "+ incorrect_label[count_incorrect_list]+ '-average_activation.png'
            plt.figure()
            plt.imshow(overlay)
            plt.savefig(os.path.join(incorrect_dir, overlay_filename))
            plt.figure()
            plt.imshow(activation_resized)
            plt.savefig(os.path.join(incorrect_dir, average_activation_filename))
            


            print("saving incorrect class image")
            print(incorrect_list_basename[count_incorrect_list])
            #shows just the activation
            # plt.imshow(activation_resized)
            # plt.show()
            
            
            count_incorrect_list = count_incorrect_list + 1
        print("stats of classified test images,top incorrect. bottom correct")
        print(count_incorrect_list)
        print(count_correct_list)


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
            if len(image_to_test)>600:
                image_to_test = image_to_test[:1005]
                test_labels = test_labels[:1005]

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


    # print(mode_to_run)





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


