#used to read in the csv file
import pandas as pd
#used to shuffle the data
from sklearn.utils import shuffle
#used for viewing images
import matplotlib.pyplot as plt
#for image loading and manipulation
import cv2
#for getting data paths
import os
#for adding command line arguments
import argparse
#easy resizing images
from PIL import Image
#model stuff
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout
#math functions
import math
#for sleep 
import time
#for getting the datetime
from datetime import datetime
#for copying files
import shutil 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt




from myModels import *
from plots import *
from utils import *
from preprocessData import *



if __name__ == "__main__":
    #inception_v3_multiple_inputs(256,256)
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S")


    parser = argparse.ArgumentParser(description='Command line tool for easily running this dataset')
    parser.add_argument("-m","--mode",default=0,help="What mode to run will run different code. 1:circle crops the images; 2:resizes the images; 3:Adds blur to the images; 4:Shows the images; 5:Trains a model; 6:Plots the accuracy; 7:Plots the loss",type=int)
    parser.add_argument("-width","--image_width", default=256,help="if resizing images what width to resize them to",type=int)
    parser.add_argument("-height", "--image_height",default=256,help="if resizing images what height to resize them to",type=int)
    parser.add_argument("-o","--output_dir",default=None,help="If running a mode the produces an output, saves the items here")
    parser.add_argument("-d","--dir",default=None,help="directory in which the imagse are located", type=str)
    parser.add_argument("-d2","--dir2",default=None,help="directory containing second dataset of images",type=str)
    parser.add_argument("-csv","--csv_location",default=None,help="location of the csv data file",type=str)
    parser.add_argument("-b","--batch_size",default=128,help="batch size for training",type=int)
    parser.add_argument("-e","--epochs",default=10000,help="Number of epochs to train the network on",type=int)
    parser.add_argument("-ti","--test_interval",default=50,help="How oftern to use test the model on the test data",type=int)
    parser.add_argument("-si","--save_interval",default=1,help="After how many epochs to save the model to a checkpoint",type=int)
    parser.add_argument("-l","--load_model",default=None,help="Option to load a saved model",type=str)
    parser.add_argument("-td","--test_data_percentage",default=0.3,help="Percentage of data to use for test data",type=float)
    parser.add_argument("-pd","--plot_dir",default=None,help="directory containing data to plot",type=str)
    parser.add_argument("-model","--model_to_use",default=1,help="Selects what model to use",type=int)
    parser.add_argument("-trainable","--trainable_transfer",default=True,help="Can the transfer learning model learn on the new data",type=bool)
    args = parser.parse_args()

    image_dir = args.dir
    csv_dir = args.csv_location
    run_mode = args.mode
    output_dir = args.output_dir
    new_image_width = args.image_width
    new_image_height = args.image_height
    batch_size = args.batch_size
    total_epochs = args.epochs
    test_interval = args.test_interval
    save_interval = args.save_interval
    model_to_load = args.load_model
    test_data_percentage = args.test_data_percentage
    plot_directory = args.plot_dir
    model_to_use = args.model_to_use
    transfer_trainable = args.trainable_transfer

    second_image_dir = args.dir2

    model_name = "None"

    if plot_directory is not None:
        plot_directory = os.path.abspath(plot_directory)

    if image_dir is None and run_mode!=6 and run_mode!=7:
        raise SyntaxError('directory for images must be provided')

    if csv_dir is None and run_mode!=6 and run_mode!=7:
        raise SyntaxError('Location for data labels csv file must be provided')



    if run_mode == 1 or run_mode == 2 or run_mode == 3:
        if output_dir is None:
            raise SyntaxError('Must provide a directory for the output')
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    

    if run_mode !=6 and run_mode!=7:
        #loads the data
        health_level,image_name = load_data(csv_dir)

        #gets the path of the data
        data_path = os.path.abspath(image_dir)
        image_name = get_full_image_name(data_path,image_name)

    #cirlce crops the images
    if run_mode == 1:
        circle_crop_v2(image_name,output_dir)
    if run_mode == 2:
        resize_image(image_name,new_image_width,new_image_height,output_dir)
    if run_mode == 3:
        #print(image_name)
        add_blur(image_name,output_dir)
    if run_mode == 4:
        show_images(image_name)
    
    if run_mode == 5:
        os.makedirs(dt_string)
        run_dir = os.path.abspath(dt_string)
        if model_to_load is None:
            #there are 30ish missing files, should make a new csv later
            health_level,image_name = remove_nonexistent_data(health_level,image_name)
            #shuffles the data to randomize starting train and test data
            health_level,image_name = shuffle_data(health_level,image_name)
            #way to many zeros in the data
            health_level,image_name = trim_data(run_dir,health_level,image_name)
            #shuffles the data to randomize starting train and test data
            health_level,image_name = shuffle_data(health_level,image_name)
            #splits the data into train and test
            train_images,train_labels,test_images,test_labels = split_data_train_test(run_dir,health_level,image_name,test_data_percentage)

            #sets the epochs and save points
            current_epoch = 0
            previous_save = 0
            
            #loads the trimmed data
            trimmed_csv_file = run_dir + "/csv_files/trimmed.csv"
            trimmed_labels,trimmed_images = load_data(trimmed_csv_file)
            #creates a new model
            if model_to_use == 1:
                model = create_CNN(new_image_width, new_image_height)
                model_name = "CNN"
            elif model_to_use == 2:
                if transfer_trainable:
                    model = transfer_learning_model_inception_v3(new_image_width, new_image_height,True)
                else:
                    model = transfer_learning_model_inception_v3(new_image_width, new_image_height,False)

                model_name = "inception_transfer"
            elif model_to_use == 3:
                model = inception_v3_multiple_inputs(new_image_width, new_image_height)
            else:
                raise SyntaxError('Not an implemented model')

        else:
            #gets the full path of the checkpoint file that is being loaded
            full_check_point_path = os.path.abspath(model_to_load)
            #creates a directory for the plots
            plots_run_dir = run_dir + "/plots"
            os.makedirs(plots_run_dir)
            #creates a directory for checkpoints
            checkpoints_run_dir = run_dir + "/checkpoints"
            os.makedirs(checkpoints_run_dir)
            #finds the directory one level up
            one_level_up_dir = get_previous_directory(full_check_point_path)

            #sets the plot dir so we can copy files
            old_plot_dir = one_level_up_dir + "/plots/"
            #copies all old files here and moves it to new directory
            files_in_old_plots = os.listdir(old_plot_dir)
            for files in files_in_old_plots:
                source_file = old_plot_dir + files
                dest_file = plots_run_dir + "/" + files
                shutil.copyfile(source_file,dest_file)

            #copies the checkpoint file to new run folder
            shutil.copyfile(full_check_point_path,checkpoints_run_dir+"/checkpoint0")

            #loads the last epoch
            epoch_file = old_plot_dir + "epochs.npy"
            current_epoch = get_last_epoch(epoch_file)
            previous_save = math.floor(current_epoch)

            #sets old csv_dir to copy
            old_csv_dir = one_level_up_dir + "/csv_files"
            #sets new csv dir
            new_csv_dir = run_dir + "/csv_files"
            #copies the old csv files to the new directory
            old_csv_files = os.listdir(old_csv_dir)
            #makes the directory if it does not exists
            os.makedirs(new_csv_dir)
            for files in old_csv_files:
                source_file = old_csv_dir + "/" + files
                dest_file = new_csv_dir + "/" + files
                shutil.copyfile(source_file,dest_file)
            
            #loads data from the csv files
            train_csv_file = new_csv_dir + "/train.csv"
            test_csv_file = new_csv_dir + "/test.csv"
            trimmed_csv_file = new_csv_dir + "/trimmed.csv"

            #loads this data into the list
            train_labels,train_images = load_data(train_csv_file)
            test_labels,test_images = load_data(test_csv_file)
            trimmed_labels,trimmed_images = load_data(trimmed_csv_file)

            #gets the full paths of the images
            train_images = get_full_image_name_no_ext(data_path,train_images)
            test_images = get_full_image_name_no_ext(data_path,test_images)
            

            #loads the saved model if needed
            model = load_model(model_to_load)


            
    #     #gets the first batch of testing data
    #     datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    #     read_images = []
    #     for image in train_images:
    #         temp = cv2.imread
    #         read_images.append(temp)
    #     print(len(read_images))

    #     datagen.fit(read_images)
    #     for image_batch, labels_batch in datagen.flow(read_images, train_labels, batch_size=9):
    #         for i in range(0, 9):
    #             plt.subplot(330 + 1 + i)
    #             plt.imshow(image_batch[i].reshape(256, 256))
	# # show the plot
    #         plt.show()
    #         break



        #model.summary()
        if model_to_use!=3:
            np_image_batch_test,test_labels_batch = prepare_data_for_model(1000,test_labels,test_images,new_image_width,new_image_height)
            total_images = len(train_images)
            images_trained_on = 0
            while current_epoch<total_epochs:
                #gets images and labels ready for model input
                np_image_batch,label_batch = prepare_data_for_model(batch_size,train_labels,train_images,new_image_width,new_image_height)
                #trains on the input
                model.train_on_batch(np_image_batch, label_batch)
                #adds the number of images to the total count
                images_trained_on = images_trained_on + batch_size
                #outputs stats after 5 batchs
                if images_trained_on % (test_interval*batch_size)==0:
                    print("Evaulating on test data...")
                    #gets the metrics for the test data
                    metrics = model.evaluate(np_image_batch_test, test_labels_batch)
                    loss_test = metrics[0]
                    accuracy_test = metrics[-1]
                    #gets the metrics for the training data
                    print("Evaluating on training data...")
                    metrics = model.evaluate(np_image_batch,label_batch)
                    loss_train = metrics[0]
                    accuracy_train = metrics[-1]
                    #adds data to numpy files
                    add_plot_data(accuracy_test,accuracy_train,loss_test,loss_train,current_epoch,run_dir)
                    #gets new dataset for testing
                    np_image_batch_test,test_labels_batch = prepare_data_for_model(1000,test_labels,test_images,new_image_width,new_image_height)
                #increments the epoch
                current_epoch = current_epoch + batch_size/total_images
                #saves the epoch if the save increment has passed
                if current_epoch - save_interval>previous_save:
                    save_model(model,run_dir)
                    previous_save = previous_save + save_interval
                print("epoch: ",current_epoch)
        #multiple inputs
        else:
            test_images_two = change_dir_name(second_image_dir,test_images)
            train_images_two = change_dir_name(second_image_dir,train_images)
            np_image_batch_test,np_image_batch_test_two,test_labels_batch = prepare_data_for_model_two(250,test_labels,test_images,test_images_two,new_image_width,new_image_height)
            #quit()
            total_images = len(train_images)
            images_trained_on = 0
            while current_epoch<total_epochs:
                #gets images and labels ready for model input
                np_image_batch,np_image_batch_two,label_batch = prepare_data_for_model_two(batch_size,train_labels,train_images,train_images_two,new_image_width,new_image_height)
                #trains on the input
                model.train_on_batch([np_image_batch,np_image_batch_two], label_batch)
                #adds the number of images to the total count
                images_trained_on = images_trained_on + batch_size
                #outputs stats after 5 batchs
                if images_trained_on % (test_interval*batch_size)==0:
                    print("Evaulating on test data...")
                    #gets the metrics for the test data
                    metrics = model.evaluate([np_image_batch_test,np_image_batch_test_two], test_labels_batch)
                    loss_test = metrics[0]
                    accuracy_test = metrics[-1]
                    #gets the metrics for the training data
                    print("Evaluating on training data...")
                    metrics = model.evaluate([np_image_batch,np_image_batch_two],label_batch)
                    loss_train = metrics[0]
                    accuracy_train = metrics[-1]
                    #adds data to numpy files
                    add_plot_data(accuracy_test,accuracy_train,loss_test,loss_train,current_epoch,run_dir)
                    #gets new dataset for testing
                    np_image_batch_test,np_image_batch_test_two,test_labels_batch = prepare_data_for_model_two(250,test_labels,test_images,test_images_two,new_image_width,new_image_height)
                #increments the epoch
                current_epoch = current_epoch + batch_size/total_images
                #saves the epoch if the save increment has passed
                if current_epoch - save_interval>previous_save:
                    save_model(model,run_dir)
                    previous_save = previous_save + save_interval
                print("epoch: ",current_epoch)

    

    if run_mode == 6:
        plot_accuracy(plot_directory)
    if run_mode == 7:
        plot_loss(plot_directory)
    