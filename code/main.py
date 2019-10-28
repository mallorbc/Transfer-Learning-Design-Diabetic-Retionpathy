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

from myModels import *
from plots import *
from utils import *
from preprocessData import *


# def save_csv(csv_list,image_name,output_dir)
#     label_dir = output_dir + "/labels"
#     if not os.path.exists(label_dir):
#         os.makedirs(label_dir)
#     for i in range(len(label_list)):
#     data = pd.DataFrame(csv_list,)



if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S")




    parser = argparse.ArgumentParser(description='Command line tool for easily running this dataset')
    parser.add_argument("-m","--mode",default=0,help="What mode to run will run different code",type=int)
    parser.add_argument("-width","--image_width", default=512,help="if resizing images what width to resize them to",type=int)
    parser.add_argument("-height", "--image_height",default=512,help="if resizing images what height to resize them to",type=int)
    parser.add_argument("-o","--output_dir",default=None,help="If running a mode the produces an output, saves the items here")
    parser.add_argument("-d","--dir",default=None,help="directory in which the imagse are located", type=str)
    parser.add_argument("-csv","--csv_location",default=None,help="location of the csv data file",type=str)
    parser.add_argument("-b","--batch_size",default=128,help="batch size for training",type=int)
    parser.add_argument("-e","--epochs",default=100,help="Number of epochs to train the network on",type=int)
    parser.add_argument("-ti","--test_interval",default=25,help="How oftern to use test the model on the test data",type=int)
    parser.add_argument("-si","--save_interval",default=1,help="After how many epochs to save the model to a checkpoint",type=int)
    parser.add_argument("-l","--load_model",default=None,help="Option to load a saved model",type=str)
    parser.add_argument("-td","--test_data_percentage",default=0.1,help="Percentage of data to use for test data",type=float)
    parser.add_argument("-pd","--plot_dir",default=None,help="directory containing data to plot",type=str)
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
    
    if plot_directory is not None:
        plot_directory = os.path.abspath(plot_directory)

    if image_dir is None and run_mode!=4:
        raise SyntaxError('directory for images must be provided')

    if csv_dir is None and run_mode!=4:
        raise SyntaxError('Location for data labels csv file must be provided')



    if run_mode == 1:
        if output_dir is None:
            raise SyntaxError('Must provide a directory for the output')
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    

    if run_mode !=4:
        #loads the data
        health_level,image_name = load_data(csv_dir)

        #gets the path of the data
        data_path = os.path.abspath(image_dir)
        image_name = get_full_image_name(data_path,image_name)

        #this shows that the data has way to many zeros
        #get_info_on_data(health_level)
        #get_info_on_data(health_level)
        # print(len(health_level))
        # print(len(image_name))
        #get_image_width_height(image_name)

    if run_mode == 1:
        resize_image(image_name,new_image_width,new_image_height,output_dir)
    if run_mode == 2:
        show_images(image_name)
    
    if run_mode == 3:
        os.makedirs(dt_string)
        run_dir = os.path.abspath(dt_string)
        #there are 30ish missing files, should make a new csv later
        health_level,image_name = remove_nonexistent_data(health_level,image_name)
        #shuffles the data to randomize starting train and test data
        health_level,image_name = shuffle_data(health_level,image_name)
        #way to many zeros in the data
        health_level,image_name = trim_data(health_level,image_name)
        #shuffles the data to randomize starting train and test data
        health_level,image_name = shuffle_data(health_level,image_name)
        #splits the data into train and test
        train_images,train_labels,test_images,test_lables = split_data_train_test(health_level,image_name,test_data_percentage)
        #normalizes the test data
        test_image_batch = normalize_images(test_images)
        np_image_batch_test = np.asarray(test_image_batch)
        #resizes the test data to fit into model
        np_image_batch_test.reshape(len(test_images),new_image_width,new_image_height,3)



        model = create_CNN(new_image_width, new_image_height)
        #model.summary()

        #if we load a model, we need to copy some files over to the new run directory
        if model_to_load is not None:
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
            print(one_level_up_dir)
            old_plot_dir = one_level_up_dir + "/plots/"
            #copies all old files here and moves it to new directory
            files_in_old_plots = os.listdir(old_plot_dir)
            for files in files_in_old_plots:
                source_file = old_plot_dir + files
                dest_file = plots_run_dir + "/" + files
                shutil.copyfile(source_file,dest_file)
            #copies the checkpoint file to new run folder
            shutil.copyfile(full_check_point_path,checkpoints_run_dir+"/checkpoint0")

        #loads a model if a flag is provided
        if model_to_load is not None:
            model = load_model(model_to_load)
            epoch_file = old_plot_dir + "epochs.npy"
            current_epoch = get_last_epoch(epoch_file)
            previous_save = math.floor(current_epoch)
        else:
            current_epoch = 0
            previous_save = 0

        
        

        image_batch = []
        label_batch = []
        total_images = len(train_images)
        images_trained_on = 0
        while current_epoch<total_epochs:
            train_labels,train_images = shuffle_data(train_labels,train_images)
            counter = 0
            for i in range(len(train_images)):
                image_batch.append(train_images[i])
                label_batch.append(train_labels[i])
                counter = counter +1
                if counter == batch_size:
                    images_trained_on = images_trained_on + batch_size
                    break
            #normalizes image pixels to 0 and 1
            image_batch = normalize_images(image_batch)
            #loads data as a np array and then reshapes it
            np_image_batch = np.asarray(image_batch)
            np_image_batch.reshape(batch_size,new_image_width,new_image_height,3)
            model.train_on_batch(np_image_batch, label_batch)
            #outputs stats after 5 batchs
            if images_trained_on % (test_interval*batch_size)==0:
                #gets the metrics
                metrics = model.evaluate(np_image_batch_test, test_lables)
                accuracy = metrics[-1]
                #adds data to numpy files
                add_plot_data(accuracy,current_epoch,run_dir)
            image_batch.clear()
            label_batch.clear()
            current_epoch = current_epoch + batch_size/total_images
            if current_epoch - save_interval>previous_save:
                save_model(model,run_dir)
                previous_save = previous_save + save_interval
            print("epoch: ",current_epoch)
    

    if run_mode == 4:
        plot_accuracy(plot_directory)
        