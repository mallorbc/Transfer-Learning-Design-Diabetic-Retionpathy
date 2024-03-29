#for getting data paths
import os
#for adding command line arguments
import argparse
#model stuff
import numpy as np
import tensorflow as tf
#for sleep 
import time
#for getting the datetime
from datetime import datetime
#for copying files
import shutil 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.utils.class_weight import compute_class_weight




from myModels import *
from plots import *
from utils import *
from preprocessData import *





if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S")


    parser = argparse.ArgumentParser(description='Command line tool for easily running this dataset')
    parser.add_argument("-n","--name",default=None,help="Name of the folder for the run",type=str)
    parser.add_argument("-m","--mode",default=0,help="What mode to run will run different code. 1:Create random train and test csv; 2:circle crops the images; 3:resizes the images; 4:Adds blur to the images; 5:Shows the images; 6:Trains a model; 7:Plots the accuracy; 8:Plots the loss; 9:Confusion Matrix",type=int)
    parser.add_argument("-width","--image_width", default=256,help="if resizing images what width to resize them to",type=int)
    parser.add_argument("-height", "--image_height",default=256,help="if resizing images what height to resize them to",type=int)
    parser.add_argument("-o","--output_dir",default=None,help="If running a mode the produces an output, saves the items here")
    parser.add_argument("-d","--dir",default=None,help="directory in which the imagse are located", type=str)
    parser.add_argument("-d2","--dir2",default=None,help="directory containing second dataset of images",type=str)
    parser.add_argument("-d3","--dir3",default=None,help="directory containing the third dataset",type=str)
    parser.add_argument("-csv","--csv_location",default=None,help="location of the csv data file",type=str)
    parser.add_argument("-b","--batch_size",default=48,help="batch size for training",type=int)
    parser.add_argument("-e","--epochs",default=500,help="Number of epochs to train the network on",type=float)
    parser.add_argument("-ti","--test_interval",default=0.5,help="How oftern to use test the model on the test data",type=float)
    parser.add_argument("-si","--save_interval",default=0.5,help="After how many epochs to save the model to a checkpoint",type=float)
    parser.add_argument("-l","--load_model",default=None,help="Option to load a saved model",type=str)
    parser.add_argument("-td","--test_data_percentage",default=0.25,help="Percentage of data to use for test data",type=float)
    parser.add_argument("-vd","--val_data_percentage",default=0.2,help="Percentage of data for validation",type=float)
    parser.add_argument("-pd","--plot_dir",default=None,help="directory containing data to plot",type=str)
    parser.add_argument("-model","--model_to_use",default=None,help="Selects what model to use",type=int)
    parser.add_argument("-trainable","--trainable_transfer",default=True,help="Can the transfer learning model learn on the new data",type=str2bool)
    parser.add_argument("-pe","--plot_epoch",default=None,help="What eopch to stop early at",type=float)
    parser.add_argument("-pes","--plot_epoch_start",default=None,help="What eopch to start plotting at",type=float)
    parser.add_argument("-np","--numpy",default=False,help="Whether the data outputed should be numpy, and whether the data loaded is numpy",type=bool)
    parser.add_argument("-mem","--gpu_mem",default=None,help="allows us to not use all the memory, useful for testing a model that is currently training",type=float)
    parser.add_argument("-train_csv",default=None,help="This allows us to specifiy what photos to use for training",type=str)
    parser.add_argument("-zca","--zca_batch_size",default=1,help="number of images per ZCA preprocessing batch",type=int)
    parser.add_argument("-blur","--blur",default=False,help="applies Gaussian blur prior to ZCA preprocessing",type=str2bool)
    parser.add_argument("-test_csv",default=None,help="This allows us to specifiy what photos to use for testing",type=str)
    parser.add_argument("-aug","--augment_data",default=True,help="flag on whether to use image data augmentation",type=str2bool)
    parser.add_argument("-test_size",default=400,help="what batch size to use for testing the performance of the models",type=int)
    parser.add_argument('-saver',"--saver_mode",default=None,help="Whether or not to save the whole model or just the weights",type=int)
    parser.add_argument('-class_size',default=708,help="how big each class should be; Can't be larger than 700",type=int)
    parser.add_argument("-plot_name",default=None,help="what to name the plots",type=str)
    parser.add_argument('-compat',"--compatibility",default=False,help="whether or not to use compat mode",type=str2bool)
    parser.add_argument("-lr","--learning_rate",default=None,help="optional flag to manually set learning rate",type=float)
    parser.add_argument("-train_method",default=1,help="How to train the model",type=int)
    parser.add_argument("-smudge_weights",default=False,help="Should the weights be not completely balanced",type=str2bool)
    parser.add_argument("-binary",default=False,help="flag for when doing things that only deal with binary classifier",type=str2bool)
    parser.add_argument("-model_binary",default=None,help="weights for binary classifier",type=str)
    parser.add_argument("-model_nonzero",default=None,help="weights for the non zero",type=str)




    
    args = parser.parse_args()

    if args.gpu_mem is not None:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))


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
    validation_data_percent = args.val_data_percentage
    plot_directory = args.plot_dir
    model_to_use = args.model_to_use
    transfer_trainable = args.trainable_transfer
    plot_epoch = args.plot_epoch
    folder_name = args.name
    data_is_numpy = args.numpy
    zca_batch_size = args.zca_batch_size        #added recently
    zca_blur = args.blur                        #added recently

    loaded_train_csv = args.train_csv
    loaded_test_csv = args.test_csv

    data_aug = args.augment_data

    saver_mode = args.saver_mode

    size_of_each_class = args.class_size

    plot_name = args.plot_name

    manual_lr = args.learning_rate

    training_method = args.train_method

    should_smudge_weights = args.smudge_weights

    binary_mode = args.binary

    binary_model_weights = args.model_binary

    non_zero_model_weights = args.model_nonzero

    plot_epoch_start = args.plot_epoch_start

    

    if folder_name is not None:
        folder_name = os.path.realpath(folder_name)

    test_size = args.test_size

    second_image_dir = args.dir2


    if plot_directory is not None:
        plot_directory = os.path.abspath(plot_directory)

    if image_dir is None and run_mode!=7 and run_mode!=8:
        raise SyntaxError('directory for images must be provided')
    
    #this variable tracks whether we are loading csv files or not
    loaded_train_test_csv = False
    if csv_dir is None and run_mode!=7 and run_mode!=8:
        if loaded_train_csv is None or loaded_test_csv is None:
            raise SyntaxError('Location for data labels csv file must be provided')
    elif csv_dir is not None and (loaded_train_csv is not None or loaded_train_csv is not None):
        # raise SyntaxError('You must either use random train test split or provide a split, not both')
        loaded_train_test_csv = True




    if run_mode == 2 or run_mode == 3 or run_mode==4 or run_mode==22:
        if output_dir is None:
            raise SyntaxError('Must provide a directory for the output')
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    

    if run_mode !=7 and run_mode!=8:
        #loads the data
        health_level,image_name = load_data(csv_dir)

        #gets the path of the data
        data_path = os.path.abspath(image_dir)
        image_name = get_full_image_name(data_path,image_name)

    if run_mode == 1 or run_mode == 6 or run_mode == 9:
        if folder_name is not None:
            os.makedirs(folder_name)
            run_dir = folder_name
        else:
            os.makedirs(dt_string)
            run_dir = os.path.abspath(dt_string)
    

    if run_mode == 1:
        #there are 30ish missing files, should make a new csv later
        health_level,image_name = remove_nonexistent_data(health_level,image_name)
        #used for cutting the classes
        health_dict = make_diagnose_class_dict(health_level,image_name)
        #cuts class 0 in half
        # health_level,image_name = cut_data_class(health_dict,0,0.5)

        #shuffles the data to randomize starting train and test data
        health_level,image_name = shuffle_data(health_level,image_name)
        #way to many zeros in the data
        # health_level,image_name = trim_data(run_dir,health_level,image_name)
        #shuffles the data to randomize starting train and test data
        # health_level,image_name = trim_data_even(run_dir,health_level,image_name,size_of_each_class)
        # health_level,image_name = shuffle_data(health_level,image_name)
        #splits the data into train and test
        train_images,train_labels,test_images,test_labels = split_data_train_test_val(run_dir,health_level,image_name,test_data_percentage,validation_data_percent)
    #cirlce crops the images

    if run_mode == 21:
        regcrop(image_name, output_dir,data_is_numpy)

    if run_mode == 22:
        zca_nocirc(image_name, output_dir, zca_batch_size, zca_blur) 

    if run_mode == 2:
        circle_crop_v2(image_name,output_dir,data_is_numpy)
    if run_mode == 3:
        resize_image(image_name,new_image_width,new_image_height,output_dir)
    if run_mode == 4:
        #print(image_name)
        add_blur(image_name,output_dir)        
    if run_mode == 5:
        show_images(image_name)
    
    if run_mode == 6 or run_mode == 9: 
        # run_dir = os.path.abspath(dt_string)
        if model_to_load is None:
            if loaded_train_test_csv is False:
                #there are 30ish missing files, should make a new csv later
                health_level,image_name = remove_nonexistent_data(health_level,image_name)
                #used for cutting the classes
                health_dict = make_diagnose_class_dict(health_level,image_name)
                #cuts class 0 in half
                # health_level,image_name = cut_data_class(health_dict,0,0.5)

                #shuffles the data to randomize starting train and test data
                health_level,image_name = shuffle_data(health_level,image_name)
                #way to many zeros in the data
                # health_level,image_name = trim_data(run_dir,health_level,image_name)
                #shuffles the data to randomize starting train and test data
                # health_level,image_name = trim_data_even(run_dir,health_level,image_name,size_of_each_class)
                # health_level,image_name = shuffle_data(health_level,image_name)
                #splits the data into train and test
                train_images,train_labels,test_images,test_labels = split_data_train_test_val(run_dir,health_level,image_name,test_data_percentage,validation_data_percent)
            #else we are loading data from other csv files
            else:
                train_labels,train_images = load_data(loaded_train_csv)
                test_labels,test_images = load_data(loaded_test_csv)
                # trimmed_labels,trimmed_images = load_data(trimmed_csv_file)

                #gets the full paths of the images
                train_images = get_full_image_name_no_ext(data_path,train_images)
                test_images = get_full_image_name_no_ext(data_path,test_images)
                #saves the csv files into the new run directory
                save_csv(run_dir,"train.csv",train_labels,train_images)
                save_csv(run_dir,"test.csv",test_labels,test_images)





            #sets the epochs and save points
            current_epoch = 0
            previous_save = 0
            #allows user to pass nothing for loading a model
            if model_to_use is None:
                model_to_use = 1
            #creates a new model
            if model_to_use == 1:
                if manual_lr is None:
                    model = create_CNN(new_image_width, new_image_height)
                else:
                    model = create_CNN(new_image_width, new_image_height,manual_lr)
            elif model_to_use == 2:
                if transfer_trainable:
                    if manual_lr is None:
                        model = transfer_learning_model_inception_v3_functional(new_image_width, new_image_height,is_trainable=True)
                    else:
                        model = transfer_learning_model_inception_v3_functional(new_image_width, new_image_height,is_trainable=True,learning_rate=manual_lr)

                else:
                    if manual_lr is None:
                        model = transfer_learning_model_inception_v3_functional(new_image_width, new_image_height,is_trainable=False)
                    else:
                        model = transfer_learning_model_inception_v3_functional(new_image_width, new_image_height,is_trainable=False,learning_rate=manual_lr)

            elif model_to_use == 3:
                model = inception_v3_multiple_inputs(new_image_width, new_image_height)
            elif model_to_use == 4:
                model = efficientnet(new_image_width, new_image_height)
            elif model_to_use == 5:
                model = efficientnet(new_image_width, new_image_height)
            elif model_to_use == 6:
                model = inception_v3_functional_binary(new_image_width, new_image_height)
            elif model_to_use == 7:
                binary_model = inception_v3_functional_binary_no_act(new_image_width,new_image_height)
                binary_model = load_model(binary_model_weights,77)
                non_zero_model = inception_v3_non_zero_no_act(new_image_width, new_image_height)
                non_zero_model = load_model(non_zero_model_weights,78)
                model = dual_inception_v3_ensemble(new_image_width, new_image_height,binary_model,non_zero_model)
            elif model_to_use == 8:
                model = inception_v3_non_zero(new_image_width, new_image_height)
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
            # one_level_up_dir = get_previous_directory(full_check_point_path)
            one_level_up_dir = get_two_directories_up(full_check_point_path)


            #sets the plot dir so we can copy files
            old_plot_dir = one_level_up_dir + "/plots/"
            #copies all old files here and moves it to new directory
            files_in_old_plots = os.listdir(old_plot_dir)
            for files in files_in_old_plots:
                source_file = old_plot_dir + files
                dest_file = plots_run_dir + "/" + files
                shutil.copyfile(source_file,dest_file)

            #copies the checkpoint file to new run folder
            # shutil.copytree(full_check_point_path,checkpoints_run_dir+"/checkpoint0")

            #loads the last epoch
            epoch_file = old_plot_dir + "epochs.npy"
            accuracy_train_file = old_plot_dir + "accuracy_train.npy"
            loss_train_file = old_plot_dir + "loss_train.npy"
            accuracy_test_file = old_plot_dir + "accuracy_test.npy"
            loss_test_file = old_plot_dir + "loss_test.npy"

            current_epoch = get_last_epoch(epoch_file)
            previous_test_epoch = current_epoch
            previous_save = current_epoch

            #realods the best values form the network
            highest_train_accuracy,highest_train_accuracy_epoch = get_highest_accuracy(epoch_file,accuracy_train_file)
            highest_test_accuracy,highest_test_accuracy_epoch = get_highest_accuracy(epoch_file,accuracy_test_file)
            lowest_train_loss,lowest_train_loss_epoch = get_lowest_loss(epoch_file,loss_train_file)
            lowest_test_loss,lowest_test_loss_epoch = get_lowest_loss(epoch_file,loss_test_file) 


            

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
            trimmed_csv_file = new_csv_dir + "/validation.csv"

            #loads this data into the list
            train_labels,train_images = load_data(train_csv_file)
            test_labels,test_images = load_data(test_csv_file)
            # trimmed_labels,trimmed_images = load_data(trimmed_csv_file)

            #gets the full paths of the images
            train_images = get_full_image_name_no_ext(data_path,train_images)
            test_images = get_full_image_name_no_ext(data_path,test_images)
            

            #loads the saved model if needed
            # print(model_to_use)
            # print(model_to_load)
            model = load_model(model_to_load,model_to_use,new_image_width,new_image_height)
        if model_to_use == 6:
            train_labels = convert_labels_to_binary(train_labels)
            test_labels = convert_labels_to_binary(test_labels)
        if model_to_use == 8:
            train_labels,train_images = convert_labels_to_non_zero(train_labels,train_images)
            test_labels,test_images = convert_labels_to_non_zero(test_labels,test_images)

        health_dict_train = make_diagnose_class_dict(train_labels,train_images)
        health_dict_test = make_diagnose_class_dict(test_labels,test_images)

        if model_to_use == 3:
            test_images_two = change_dir_name(second_image_dir,test_images)
            train_images_two = change_dir_name(second_image_dir,train_images)
            health_dict_train2 = make_diagnose_class_dict(train_labels,train_images_two)
            health_dict_test2 = make_diagnose_class_dict(test_labels,test_images_two)




            

        #model.summary()



        #value used to test the network every test interval
        if model_to_load is None:
            previous_test_epoch = 0.0
            highest_train_accuracy = 0.0
            highest_train_accuracy_epoch = 0.0
            lowest_train_loss = float("inf")
            lowest_train_loss_epoch = 0.0
            highest_test_accuracy = 0.0
            highest_test_accuracy_epoch = 0.0
            lowest_test_loss = float("inf")
            lowest_test_loss_epoch = 0.0
        #used to limit printing
        next_print = 0.00
        total_images = len(train_images)
        # print(train_labels)
        # quit()

        images_trained_on = 0

        if data_aug is True:
            print("Using data augmenation")
            datagen_train = ImageDataGenerator(
            rotation_range=360,
            zoom_range=0.1,
            rescale = 1.0/255.0,
            shear_range=0.2,
            brightness_range=[0.8,1.2],
            height_shift_range=0.1,
            width_shift_range=0.1)
            train_images = np.asarray(train_images)
            train_labels = np.asarray(train_labels)
            # train_labels = pd.DataFrame(train_images)
            # data_frame_data = [train_images,train_labels]
            # df = pd.DataFrame(data_frame_data,columns=["image","label"])
            # df = pd.DataFrame({"image": train_images,
            #                     "label": train_labels})
        
        total_images = len(train_images)
        if model_to_use !=6 and model_to_use!=8:
            print()
            print("Using " + str(total_images) + " images for training")
            print()
            print("Using " + str(len(health_dict_train[0])) + " images from class 0")
            print()
            print("Using " + str(len(health_dict_train[1])) + " images from class 1")
            print()
            print("Using " + str(len(health_dict_train[2])) + " images from class 2")
            print()
            print("Using " + str(len(health_dict_train[3])) + " images from class 3")
            print()
            print("Using " + str(len(health_dict_train[4])) + " images from class 4")
            time.sleep(3)

            # cat0_images,cat1_images,cat2_images,cat3_images,cat4_images = get_info_on_data(train_labels)	


            class_weights_dict = compute_class_weight("balanced",np.unique(train_labels),train_labels)
            class_weights_list = list(class_weights_dict)
            if should_smudge_weights is True:
                class_weights_adjust = np.asarray(class_weights_list)	
                smudge_factor = 1.0	
                class_weights_adjust = np.add(class_weights_adjust,smudge_factor)	
                class_weights_adjust = np.divide(class_weights_adjust,smudge_factor+1.0)	
                class_weights_list = class_weights_adjust

            class_weights = {0: class_weights_list[0], 1: class_weights_list[1], 2: class_weights_list[2], 3: class_weights_list[3], 4: class_weights_list[4]}
            base_weight0 = class_weights_list[0]/5
            base_weight1 = class_weights_list[1]/5
            base_weight2 = class_weights_list[2]/5
            base_weight3 = class_weights_list[3]/5
            base_weight4 = class_weights_list[4]/5

            print(class_weights)
            if model_to_use == 999 or model_to_load is not None:
                test_loss0,test_loss1,test_loss2,test_loss3,test_loss4,test_acc0,test_acc1,test_acc2,test_acc3,test_acc4 = get_loss_acc_of_each_class(model,train_images,new_image_width,new_image_height,test_size,health_dict_train)
                class_weights = adjust_base_weights2(test_loss0,test_loss1,test_loss2,test_loss3,test_loss4,base_weight0,base_weight1,base_weight2,base_weight3,base_weight4,run_dir)
                print(class_weights)
                print("intial loss: ",test_loss0,test_loss1,test_loss2,test_loss3,test_loss4)
                time.sleep(2)


        time.sleep(2)
        # quit()
        print()
        if training_method == 1:
            print("Using imbalenced data batches")
        else:
            print("Using balenced classes")
        # quit()


        start_time = time.time()
        time_array = []
        time_array = np.asarray(time_array)
        if len(test_images)<test_size:
            test_size = len(test_images)
            
        if model_to_use!=3 and model_to_use!=6 and model_to_use!=8 and run_mode == 6:
            if data_aug:
                if training_method == 2:
                    df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                else:
                    df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                df_length = df.shape[0]
                df_images = 0
            # np_image_batch_test,test_labels_batch = prepare_data_for_model_even(test_size,test_labels,test_images,new_image_width,new_image_height,health_dict_test)

            while current_epoch<total_epochs:
                if not data_aug:
                    #gets images and labels ready for model input
                    np_image_batch,label_batch = prepare_data_for_model(batch_size,train_labels,train_images,new_image_width,new_image_height,health_dict_train,training_method)

                    #trains on the input
                    if training_method == 1:
                        model.train_on_batch(np_image_batch, label_batch,class_weight=class_weights)
                    else:
                        model.train_on_batch(np_image_batch, label_batch)
                    images_trained_on = images_trained_on + batch_size
                    if next_print <= current_epoch:
                        print("epoch: % .2f , train loss: % .4f , % 0.2f, train acc: % .4f , % .2f, test loss: % .4f , % .2f, test acc: % .4f, % .2f " % (current_epoch, lowest_train_loss,lowest_train_loss_epoch,highest_train_accuracy,highest_train_accuracy_epoch,lowest_test_loss,lowest_test_loss_epoch,highest_test_accuracy,highest_test_accuracy_epoch))
                        # tracker.print_diff()
                        next_print = next_print + 0.01
                else:
                    for x_batch, y_batch in datagen_train.flow_from_dataframe(dataframe=df,x_col="image",y_col="label",target_size=(new_image_width, new_image_height),class_mode="raw", batch_size=args.batch_size):
                        if training_method == 1:
                            model.train_on_batch(x_batch,y_batch,class_weight=class_weights)
                        else:
                            model.train_on_batch(x_batch,y_batch)
                        images_trained_on = images_trained_on + batch_size
                        df_images = df_images + batch_size
                        current_epoch = current_epoch + batch_size/total_images
                        if next_print <= current_epoch:
                            print("epoch: % .2f , train loss: % .4f , % 0.2f, train acc: % .4f , % .2f, test loss: % .4f , % .2f, test acc: % .4f, % .2f " % (current_epoch, lowest_train_loss,lowest_train_loss_epoch,highest_train_accuracy,highest_train_accuracy_epoch,lowest_test_loss,lowest_test_loss_epoch,highest_test_accuracy,highest_test_accuracy_epoch))
                            next_print = next_print + 0.01
                        if ((current_epoch - previous_test_epoch)>= test_interval) or ((current_epoch - save_interval)>=previous_save):
                            # tracker.print_diff()
                            break
                        if df_images>df_length:
                            df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                            df_length = df.shape[0]
                            df_images = 0
                            break


                #adds the number of images to the total count
                #outputs stats after every test interval passes
                if (current_epoch - previous_test_epoch)>= test_interval:
                    print("Evaulating on test data...")
                    #gets the metrics for the test data
                    # metrics = model.evaluate(np_image_batch_test, test_labels_batch,verbose=0)
                    # np_image_batch_test = None
                    # test_labels_batch = None
                    loss_test,accuracy_test = evaluate_all_images(model,test_images,test_labels,test_size)
                    # loss_test = metrics[0]
                    # accuracy_test = metrics[-1]
                    # print("New test loss: ",loss_test," New test acc: ",accuracy_test)
        
                    # time.sleep(2)
                    #gets the metrics for the training data
                    print("Evaluating on training data...")
                    # np_image_batch,label_batch = prepare_data_for_model_rand(test_size,train_labels,train_images,new_image_width,new_image_height)
                    # np_image_batch,label_batch = prepare_data_for_model_even(test_size,train_labels,train_images,new_image_width,new_image_height,health_dict_train)

                    # metrics = model.evaluate(np_image_batch,label_batch,verbose=0)
                    # loss_train,accuracy_train = evaluate_all_images(model,train_images,train_labels,test_size)
                    np_image_batch = None
                    label_batch = None
                    # test_loss0,test_loss1,test_loss2,test_loss3,test_loss4,test_acc0,test_acc1,test_acc2,test_acc3,test_acc4 = get_loss_acc_of_each_class(model,train_images,new_image_width,new_image_height,test_size,health_dict_train)
                    loss_train,accuracy_train,test_loss0,test_acc0,test_loss1,test_acc1,test_loss2,test_acc2,test_loss3,test_acc3,test_loss4,test_acc4 = evaluate_all_images_each_class(model,health_dict_train,test_size)
                    add_class_acc_data(test_acc0,test_acc1,test_acc2,test_acc3,test_acc4,run_dir)
                    add_class_loss_data(test_loss0,test_loss1,test_loss2,test_loss3,test_loss4,run_dir)
                    print("Class losses: ",test_loss0,test_loss1,test_loss2,test_loss3,test_loss4)
                    print("Class accuracy: ",test_acc0,test_acc1,test_acc2,test_acc3,test_acc4)
                    class_weights = adjust_base_weights2(test_loss0,test_loss1,test_loss2,test_loss3,test_loss4,base_weight0,base_weight1,base_weight2,base_weight3,base_weight4,run_dir)
                    print("New train loss: ",loss_train," New train acc: ",accuracy_train)
                    print("New test loss: ",loss_test," New test acc: ",accuracy_test)

                    # loss_train,accuracy_train = evaluate_all_images(model,train_images,train_labels,test_size)

                    # loss_train = metrics[0]
                    # accuracy_train = metrics[-1]
                    
                    #used to print out best results in terminal
                    if loss_train<lowest_train_loss:
                        lowest_train_loss = loss_train
                        lowest_train_loss_epoch = previous_test_epoch + test_interval
                    if accuracy_train>highest_train_accuracy:
                        highest_train_accuracy = accuracy_train
                        highest_train_accuracy_epoch = previous_test_epoch + test_interval
                    if loss_test<lowest_test_loss:
                        lowest_test_loss = loss_test
                        lowest_test_loss_epoch = previous_test_epoch + test_interval
                    if accuracy_test>highest_test_accuracy:
                        highest_test_accuracy = accuracy_test
                        highest_test_accuracy_epoch = previous_test_epoch + test_interval

                    #adds data to numpy files
                    add_plot_data(accuracy_test,accuracy_train,loss_test,loss_train,current_epoch,run_dir)
                    #gets new dataset for testing
                    # np_image_batch_test,test_labels_batch = prepare_data_for_model_rand(test_size,test_labels,test_images,new_image_width,new_image_height)
                    # np_image_batch_test,test_labels_batch = prepare_data_for_model_even(test_size,test_labels,test_images,new_image_width,new_image_height,health_dict_test)

                    #updates the test interval
                    previous_test_epoch = previous_test_epoch + test_interval
                    # if previous_test_epoch
                #increments the epoch
                current_epoch = current_epoch + batch_size/total_images
                #saves the epoch if the save increment has passed
                if (current_epoch - save_interval)>=previous_save:
                    save_model(model,run_dir,saver_mode)
                    new_time = time.time()-start_time
                    time_array = np.append(time_array,new_time)
                    np.save(run_dir+"/time.npy",time_array)
                    previous_save = previous_save + save_interval
            save_model(model,run_dir,1)
            print("Finished Training")





        if model_to_use == 6 and run_mode == 6:
            print("Using " + str(total_images) + " images for training")
            print()
            print("Using " + str(len(health_dict_train[0])) + " images from class 0")
            print()
            print("Using " + str(len(health_dict_train[1])) + " images from class 1")
            class_weights_dict = compute_class_weight("balanced",np.unique(train_labels),train_labels)
            class_weights_list = list(class_weights_dict)
            # class_weights_list[0] = class_weights_list[0]/2
            # class_weights_list[1] = class_weights_list[1]/2


            class_weights = {0: class_weights_list[0], 1: class_weights_list[1]}
            base_weight0 = class_weights_list[0]
            base_weight1 = class_weights_list[1]
            base_weight0 = base_weight0/2
            base_weight1 = base_weight1/2
            print(class_weights)
            np_image_batch_test,test_labels_batch = prepare_data_for_model_even_binary(test_size,test_labels,test_images,new_image_width,new_image_height,health_dict_test)


            if data_aug:
                if training_method == 2:
                    df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                else:
                    df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                df_length = df.shape[0]
                df_images = 0
            np_image_batch_test,test_labels_batch = prepare_data_for_model_even_binary(test_size,test_labels,test_images,new_image_width,new_image_height,health_dict_test)

            while current_epoch<total_epochs:
                if not data_aug:
                    #gets images and labels ready for model input
                    np_image_batch,label_batch = prepare_data_for_model(batch_size,train_labels,train_images,new_image_width,new_image_height,health_dict_train,training_method)

                    #trains on the input
                    if training_method == 1:
                        model.train_on_batch(np_image_batch, label_batch,class_weight=class_weights)
                    else:
                        model.train_on_batch(np_image_batch, label_batch)
                    images_trained_on = images_trained_on + batch_size
                    if next_print <= current_epoch:
                        print("epoch: % .2f , train loss: % .4f , % 0.2f, train acc: % .4f , % .2f, test loss: % .4f , % .2f, test acc: % .4f, % .2f " % (current_epoch, lowest_train_loss,lowest_train_loss_epoch,highest_train_accuracy,highest_train_accuracy_epoch,lowest_test_loss,lowest_test_loss_epoch,highest_test_accuracy,highest_test_accuracy_epoch))
                        # tracker.print_diff()
                        next_print = next_print + 0.01
                else:
                    for x_batch, y_batch in datagen_train.flow_from_dataframe(dataframe=df,x_col="image",y_col="label",target_size=(new_image_width, new_image_height),class_mode="raw", batch_size=args.batch_size):
                        if training_method == 1:
                            model.train_on_batch(x_batch,y_batch,class_weight=class_weights)
                        else:
                            model.train_on_batch(x_batch,y_batch)
                        images_trained_on = images_trained_on + batch_size
                        df_images = df_images + batch_size
                        current_epoch = current_epoch + batch_size/total_images
                        if next_print <= current_epoch:
                            print("epoch: % .2f , train loss: % .4f , % 0.2f, train acc: % .4f , % .2f, test loss: % .4f , % .2f, test acc: % .4f, % .2f " % (current_epoch, lowest_train_loss,lowest_train_loss_epoch,highest_train_accuracy,highest_train_accuracy_epoch,lowest_test_loss,lowest_test_loss_epoch,highest_test_accuracy,highest_test_accuracy_epoch))
                            next_print = next_print + 0.01
                        if ((current_epoch - previous_test_epoch)>= test_interval) or ((current_epoch - save_interval)>=previous_save):
                            # tracker.print_diff()
                            break
                        if df_images>df_length:
                            df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                            df_length = df.shape[0]
                            df_images = 0
                            break


                #adds the number of images to the total count
                #outputs stats after every test interval passes
                if (current_epoch - previous_test_epoch)>= test_interval:
                    print("Evaulating on test data...")
                    #gets the metrics for the test data
                    # metrics = model.evaluate(np_image_batch_test, test_labels_batch,verbose=0)
                    np_image_batch_test = None
                    test_labels_batch = None
                    loss_test,accuracy_test = evaluate_all_images(model,test_images,test_labels,test_size)
                    # loss_test = metrics[0]
                    # accuracy_test = metrics[-1]
        
                    # time.sleep(2)
                    #gets the metrics for the training data
                    print("Evaluating on training data...")
                    # np_image_batch,label_batch = prepare_data_for_model_rand(test_size,train_labels,train_images,new_image_width,new_image_height)
                    # np_image_batch,label_batch = prepare_data_for_model_even_binary(test_size,train_labels,train_images,new_image_width,new_image_height,health_dict_train)

                    # metrics = model.evaluate(np_image_batch,label_batch,verbose=0)
                    # loss_train,accuracy_train = evaluate_all_images(model,train_images,train_labels,test_size)
                    # loss_train = metrics[0]
                    # accuracy_train = metrics[-1]
                    np_image_batch = None
                    label_batch = None
                    # test_loss0,test_loss1,test_acc0,test_acc1 = get_loss_acc_of_each_class_binary2(model,train_images,new_image_width,new_image_height,test_size,health_dict_train)
                    loss_train,accuracy_train,test_loss0,test_acc0,test_loss1,test_acc1,_,_,_,_,_,_ = evaluate_all_images_each_class(model,health_dict_train,test_size)

                    add_class_acc_data_binary(test_acc0,test_acc1,run_dir)
                    print("Class losses: ",test_loss0,test_loss1)
                    print("Class accuracy: ",test_acc0,test_acc1)
                    class_weights = adjust_base_weights_binary(test_loss0,test_loss1,base_weight0,base_weight1,run_dir)
                    add_class_loss_data_binary(test_loss0,test_loss1,run_dir)
                    print("New train loss: ",loss_train," New train acc: ",accuracy_train)
                    print("New test loss: ",loss_test," New test acc: ",accuracy_test)

                    

                    
                    #used to print out best results in terminal
                    if loss_train<lowest_train_loss:
                        lowest_train_loss = loss_train
                        lowest_train_loss_epoch = previous_test_epoch + test_interval
                    if accuracy_train>highest_train_accuracy:
                        highest_train_accuracy = accuracy_train
                        highest_train_accuracy_epoch = previous_test_epoch + test_interval
                    if loss_test<lowest_test_loss:
                        lowest_test_loss = loss_test
                        lowest_test_loss_epoch = previous_test_epoch + test_interval
                    if accuracy_test>highest_test_accuracy:
                        highest_test_accuracy = accuracy_test
                        highest_test_accuracy_epoch = previous_test_epoch + test_interval

                    #adds data to numpy files
                    add_plot_data(accuracy_test,accuracy_train,loss_test,loss_train,current_epoch,run_dir)
                    #gets new dataset for testing
                    # np_image_batch_test,test_labels_batch = prepare_data_for_model_rand(test_size,test_labels,test_images,new_image_width,new_image_height)
                    np_image_batch_test,test_labels_batch = prepare_data_for_model_even_binary(test_size,test_labels,test_images,new_image_width,new_image_height,health_dict_test)

                    #updates the test interval
                    previous_test_epoch = previous_test_epoch + test_interval
                    # if previous_test_epoch
                #increments the epoch
                current_epoch = current_epoch + batch_size/total_images
                #saves the epoch if the save increment has passed
                if (current_epoch - save_interval)>=previous_save:
                    save_model(model,run_dir,saver_mode)
                    new_time = time.time()-start_time
                    time_array = np.append(time_array,new_time)
                    np.save(run_dir+"/time.npy",time_array)
                    previous_save = previous_save + save_interval
            save_model(model,run_dir,1)
            print("Finished Training")





        if model_to_use == 8 and run_mode == 6:
            print()
            print("Using " + str(total_images) + " images for training")
            print()
            print("Using " + str(len(health_dict_train[0])) + " images from class 0")
            print()
            print("Using " + str(len(health_dict_train[1])) + " images from class 1")
            print()
            print("Using " + str(len(health_dict_train[2])) + " images from class 2")
            print()
            print("Using " + str(len(health_dict_train[3])) + " images from class 3")
            time.sleep(3)


            class_weights_dict = compute_class_weight("balanced",np.unique(train_labels),train_labels)
            class_weights_list = list(class_weights_dict)
            if should_smudge_weights is True:
                class_weights_adjust = np.asarray(class_weights_list)	
                smudge_factor = 1.0	
                class_weights_adjust = np.add(class_weights_adjust,smudge_factor)	
                class_weights_adjust = np.divide(class_weights_adjust,smudge_factor+1.0)	
                class_weights_list = class_weights_adjust

            class_weights = {0: class_weights_list[0], 1: class_weights_list[1], 2: class_weights_list[2], 3: class_weights_list[3]}

            base_weight0 = class_weights_list[0]/4
            base_weight1 = class_weights_list[1]/4
            base_weight2 = class_weights_list[2]/4
            base_weight3 = class_weights_list[3]/4

            print(class_weights)

            if model_to_use == 999 or model_to_load is not None:
                loss_train,accuracy_train,test_loss0,test_acc0,test_loss1,test_acc1,test_loss2,test_acc2,test_loss3,test_acc3,_,_ = evaluate_all_images_each_class(model,health_dict_train,test_size)
                class_weights = adjust_base_weights_test(test_loss0,base_weight0,test_loss1,base_weight1,run_dir,test_loss2,base_weight2,test_loss3,base_weight3)
                print(class_weights)
                print("intial loss: ",test_loss0,test_loss1,test_loss2,test_loss3)
                time.sleep(2)
            # np_image_batch_test,test_labels_batch = prepare_data_for_model_even_binary(test_size,test_labels,test_images,new_image_width,new_image_height,health_dict_test)


            if data_aug:
                if training_method == 2:
                    df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                else:
                    df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                df_length = df.shape[0]
                df_images = 0
            # np_image_batch_test,test_labels_batch = prepare_data_for_model_even_binary(test_size,test_labels,test_images,new_image_width,new_image_height,health_dict_test)

            while current_epoch<total_epochs:
                if not data_aug:
                    #gets images and labels ready for model input
                    np_image_batch,label_batch = prepare_data_for_model(batch_size,train_labels,train_images,new_image_width,new_image_height,health_dict_train,training_method)

                    #trains on the input
                    if training_method == 1:
                        model.train_on_batch(np_image_batch, label_batch,class_weight=class_weights)
                    else:
                        model.train_on_batch(np_image_batch, label_batch)
                    images_trained_on = images_trained_on + batch_size
                    if next_print <= current_epoch:
                        print("epoch: % .2f , train loss: % .4f , % 0.2f, train acc: % .4f , % .2f, test loss: % .4f , % .2f, test acc: % .4f, % .2f " % (current_epoch, lowest_train_loss,lowest_train_loss_epoch,highest_train_accuracy,highest_train_accuracy_epoch,lowest_test_loss,lowest_test_loss_epoch,highest_test_accuracy,highest_test_accuracy_epoch))
                        # tracker.print_diff()
                        next_print = next_print + 0.01
                else:
                    for x_batch, y_batch in datagen_train.flow_from_dataframe(dataframe=df,x_col="image",y_col="label",target_size=(new_image_width, new_image_height),class_mode="raw", batch_size=args.batch_size):
                        if training_method == 1:
                            model.train_on_batch(x_batch,y_batch,class_weight=class_weights)
                        else:
                            model.train_on_batch(x_batch,y_batch)
                        images_trained_on = images_trained_on + batch_size
                        df_images = df_images + batch_size
                        current_epoch = current_epoch + batch_size/total_images
                        if next_print <= current_epoch:
                            print("epoch: % .2f , train loss: % .4f , % 0.2f, train acc: % .4f , % .2f, test loss: % .4f , % .2f, test acc: % .4f, % .2f " % (current_epoch, lowest_train_loss,lowest_train_loss_epoch,highest_train_accuracy,highest_train_accuracy_epoch,lowest_test_loss,lowest_test_loss_epoch,highest_test_accuracy,highest_test_accuracy_epoch))
                            next_print = next_print + 0.01
                        if ((current_epoch - previous_test_epoch)>= test_interval) or ((current_epoch - save_interval)>=previous_save):
                            # tracker.print_diff()
                            break
                        if df_images>df_length:
                            df = generate_dataframe(train_images,train_labels,health_dict_train,training_method)
                            df_length = df.shape[0]
                            df_images = 0
                            break


                #adds the number of images to the total count
                #outputs stats after every test interval passes
                if (current_epoch - previous_test_epoch)>= test_interval:
                    print("Evaulating on test data...")
                    #gets the metrics for the test data
                    # metrics = model.evaluate(np_image_batch_test, test_labels_batch,verbose=0)
                    np_image_batch_test = None
                    test_labels_batch = None
                    loss_test,accuracy_test = evaluate_all_images(model,test_images,test_labels,test_size)
                    # loss_test = metrics[0]
                    # accuracy_test = metrics[-1]
        
                    # time.sleep(2)
                    #gets the metrics for the training data
                    print("Evaluating on training data...")
                    # np_image_batch,label_batch = prepare_data_for_model_rand(test_size,train_labels,train_images,new_image_width,new_image_height)
                    # np_image_batch,label_batch = prepare_data_for_model_even_binary(test_size,train_labels,train_images,new_image_width,new_image_height,health_dict_train)

                    # metrics = model.evaluate(np_image_batch,label_batch,verbose=0)
                    # loss_train,accuracy_train = evaluate_all_images(model,train_images,train_labels,test_size)
                    # loss_train = metrics[0]
                    # accuracy_train = metrics[-1]
                    np_image_batch = None
                    label_batch = None
                    loss_train,accuracy_train,test_loss0,test_acc0,test_loss1,test_acc1,test_loss2,test_acc2,test_loss3,test_acc3,_,_ = evaluate_all_images_each_class(model,health_dict_train,test_size)

                    # test_loss0,test_loss1,test_loss2,test_loss3,test_acc0,test_acc1,test_acc2,test_acc3 = get_loss_acc_of_each_class_non_zero(model,train_images,new_image_width,new_image_height,test_size,health_dict_train)
                    add_class_acc_data2(test_acc0,test_acc1,run_dir,test_acc2,test_acc3)
                    print("Class losses: ",test_loss0,test_loss1,test_loss2,test_loss3)
                    print("Class accuracy: ",test_acc0,test_acc1,test_acc2,test_acc3)
                    class_weights = adjust_base_weights_test(test_loss0,base_weight0,test_loss1,base_weight1,run_dir,test_loss2,base_weight2,test_loss3,base_weight3)
                    add_class_loss_data2(test_loss0,test_loss1,run_dir,test_loss2,test_loss3)
                    print("New train loss: ",loss_train," New train acc: ",accuracy_train)
                    print("New test loss: ",loss_test," New test acc: ",accuracy_test)

                    

                    
                    #used to print out best results in terminal
                    if loss_train<lowest_train_loss:
                        lowest_train_loss = loss_train
                        lowest_train_loss_epoch = previous_test_epoch + test_interval
                    if accuracy_train>highest_train_accuracy:
                        highest_train_accuracy = accuracy_train
                        highest_train_accuracy_epoch = previous_test_epoch + test_interval
                    if loss_test<lowest_test_loss:
                        lowest_test_loss = loss_test
                        lowest_test_loss_epoch = previous_test_epoch + test_interval
                    if accuracy_test>highest_test_accuracy:
                        highest_test_accuracy = accuracy_test
                        highest_test_accuracy_epoch = previous_test_epoch + test_interval

                    #adds data to numpy files
                    add_plot_data(accuracy_test,accuracy_train,loss_test,loss_train,current_epoch,run_dir)
                    #gets new dataset for testing
                    # np_image_batch_test,test_labels_batch = prepare_data_for_model_rand(test_size,test_labels,test_images,new_image_width,new_image_height)
                    # np_image_batch_test,test_labels_batch = prepare_data_for_model_even_binary(test_size,test_labels,test_images,new_image_width,new_image_height,health_dict_test)

                    #updates the test interval
                    previous_test_epoch = previous_test_epoch + test_interval
                    # if previous_test_epoch
                #increments the epoch
                current_epoch = current_epoch + batch_size/total_images
                #saves the epoch if the save increment has passed
                if (current_epoch - save_interval)>=previous_save:
                    save_model(model,run_dir,saver_mode)
                    new_time = time.time()-start_time
                    time_array = np.append(time_array,new_time)
                    np.save(run_dir+"/time.npy",time_array)
                    previous_save = previous_save + save_interval
            save_model(model,run_dir,1)
            print("Finished Training")


       
        #multiple inputs
        elif run_mode == 6 and model_to_use == 3:
            # test_images_two = change_dir_name(second_image_dir,test_images)
            # train_images_two = change_dir_name(second_image_dir,train_images)
            # np_image_batch_test,np_image_batch_test_two,test_labels_batch = prepare_data_for_model_two(test_size,test_labels,test_images,test_images_two,new_image_width,new_image_height)
            #quit()
            total_images = len(train_images)
            images_trained_on = 0
            while current_epoch<total_epochs:
                #gets images and labels ready for model input
                np_image_batch,np_image_batch_two,label_batch = prepare_data_for_model_two(batch_size,train_labels,train_images,train_images_two,new_image_width,new_image_height)
                #trains on the input
                if not data_aug:
                    if training_method == 1:
                        model.train_on_batch([np_image_batch,np_image_batch_two], label_batch,class_weight=class_weights)
                    else:
                        model.train_on_batch([np_image_batch,np_image_batch_two], label_batch)
                else:
                    model = dual_input_aug_train(model,train_labels,train_images,train_images_two,class_weights,datagen_train,batch_size)
                #adds the number of images to the total count
                images_trained_on = images_trained_on + batch_size
                #outputs stats after 5 batchs
                if (current_epoch - previous_test_epoch)>= test_interval:
                    
                    #gets the metrics for the train data
                    print("Evaluating on training data...")
                    loss_train,accuracy_train,test_loss0,test_acc0,test_loss1,test_acc1,test_loss2,test_acc2,test_loss3,test_acc3,test_loss4,test_acc4 = evaluate_all_images_each_class_two_inputs(model,health_dict_train,health_dict_train2,test_size)
                    print("Evaulating on test data...")
                    loss_test,accuracy_test = evaluate_all_images_two_inputs(model,test_images,test_images_two,test_labels,test_size)
                    add_class_acc_data(test_acc0,test_acc1,test_acc2,test_acc3,test_acc4,run_dir)
                    add_class_loss_data(test_loss0,test_loss1,test_loss2,test_loss3,test_loss4,run_dir)
                    add_plot_data(accuracy_test,accuracy_train,loss_test,loss_train,current_epoch,run_dir)
                    print("Class losses: ",test_loss0,test_loss1,test_loss2,test_loss3,test_loss4)
                    print("Class accuracy: ",test_acc0,test_acc1,test_acc2,test_acc3,test_acc4)
                    class_weights = adjust_base_weights2(test_loss0,test_loss1,test_loss2,test_loss3,test_loss4,base_weight0,base_weight1,base_weight2,base_weight3,base_weight4,run_dir)
                    print("New train loss: ",loss_train," New train acc: ",accuracy_train)
                    print("New test loss: ",loss_test," New test acc: ",accuracy_test)
                    


                    #used to print out best results in terminal
                    if loss_train<lowest_train_loss:
                        lowest_train_loss = loss_train
                        lowest_train_loss_epoch = previous_test_epoch + test_interval
                    if accuracy_train>highest_train_accuracy:
                        highest_train_accuracy = accuracy_train
                        highest_train_accuracy_epoch = previous_test_epoch + test_interval
                    if loss_test<lowest_test_loss:
                        lowest_test_loss = loss_test
                        lowest_test_loss_epoch = previous_test_epoch + test_interval
                    if accuracy_test>highest_test_accuracy:
                        highest_test_accuracy = accuracy_test
                        highest_test_accuracy_epoch = previous_test_epoch + test_interval

                    np_image_batch = None
                    np_image_batch_two = None
                    #adds data to numpy files
                    #gets new dataset for testing
                    # np_image_batch_test,np_image_batch_test_two,test_labels_batch = prepare_data_for_model_two(test_size,test_labels,test_images,test_images_two,new_image_width,new_image_height)
                    #updates the previous_test_epoch
                    previous_test_epoch = previous_test_epoch + test_interval
                #increments the epoch
                current_epoch = current_epoch + batch_size/total_images
                #saves the epoch if the save increment has passed
                if current_epoch - save_interval>previous_save:
                    save_model(model,run_dir,saver_mode)
                    previous_save = previous_save + save_interval
                    new_time = time.time()-start_time
                    time_array = np.append(time_array,new_time)
                    np.save(run_dir+"/time.npy",time_array)
                print("epoch: % .2f , train loss: % .4f , % 0.2f, train acc: % .4f , % .2f, test loss: % .4f , % .2f, test acc: % .4f, % .2f " % (current_epoch, lowest_train_loss,lowest_train_loss_epoch,highest_train_accuracy,highest_train_accuracy_epoch,lowest_test_loss,lowest_test_loss_epoch,highest_test_accuracy,highest_test_accuracy_epoch))
            #will save the entire model
            save_model(model,run_dir,1)
            print("Finished Training")


    

    if run_mode == 7:
        plot_accuracy(plot_directory,plot_epoch,plot_epoch_start)
        if not binary_mode:
            plot_class_acc(plot_directory,plot_epoch)
        else:
            plot_class_acc_binary(plot_directory,plot_epoch)



    if run_mode == 8:
        plot_loss(plot_directory,plot_epoch,plot_epoch_start)
        if not binary_mode:
            plot_class_losses(plot_directory,plot_epoch)
        else:
            plot_class_losses_binary(plot_directory,plot_epoch)

    
    if run_mode == 9:
        print("make matrix")
        if model_to_use !=3:
            create_confusion_matrix_one_input(model,test_images,test_labels)
        else:
            test_images_two = change_dir_name(second_image_dir,test_images)
            # train_images_two = change_dir_name(second_image_dir,train_images)
            create_confusion_matrix_two_inputs(model,test_images,test_images_two,test_labels)
    