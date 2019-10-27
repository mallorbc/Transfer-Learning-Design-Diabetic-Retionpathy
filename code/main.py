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




def load_data(data_csv):

    #reads in the data
    data = pd.read_csv(data_csv)

    #gets the columns we need
    image_name = data["image"]
    health_level = data["level"]

    #converts the data we need into a list
    health_level = list(health_level)
    image_name = list(image_name)
    return health_level,image_name

def shuffle_data(list_of_labels,list_of_image_name):
    #shuffles the data together
    health_level,image_name = shuffle(list_of_labels,list_of_image_name)
    return health_level,image_name


def show_images(image_name):
    #loops through all images
    for image in image_name:
        img = cv2.imread(image)
        #im_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

def get_info_on_data(list_of_data):
    counter = [0,0,0,0,0]
    for item in list_of_data:
        counter[item] = counter[item] +1
    print(counter)
    cat0 = counter[0]
    cat1 = counter[1]
    cat2 =  counter[2]
    cat3 = counter[3]
    cat4 = counter[4]
    return cat0,cat1,cat2,cat3,cat4

#this cuts down on the number of zeros
def trim_data(list_of_health_data,list_of_image_name):
    cat0,cat1,cat2,cat3,cat4 = get_info_on_data(list_of_health_data)
    new_list_of_health_data = []
    new_list_of_images = []
    discarded_image_names = []
    discarded_health_data = []

    cat0_counter = 0
    for i in range(len(list_of_health_data)):
        if list_of_health_data[i] !=0:
            new_list_of_health_data.append(list_of_health_data[i])
            new_list_of_images.append(list_of_image_name[i])
            cat0_counter = cat0_counter +1

        elif list_of_health_data[i] == 0 and cat0_counter %5 == 0:
            new_list_of_health_data.append(list_of_health_data[i])
            new_list_of_images.append(list_of_image_name[i])
        
        elif list_of_health_data[i] == 0 and cat0_counter % 5 != 0:
            discarded_image_names.append(list_of_image_name[i])
            discarded_health_data.append(list_of_health_data[i])

    return new_list_of_health_data,new_list_of_images

    # return new_list_of_health_data,new_list_of_images,discarded_image_names,discarded_health_data
    

def get_image_width_height(list_of_image_name):
    min_width = float("inf")
    min_height = float("inf")
    for image in list_of_image_name:
        with Image.open(image) as img:
            width ,height = img.size
        if width <min_width:
            min_width = width
        if height < min_height:
            min_height = height
        print(width,height)
        print(width/height)
    print(min_width,min_height)

def get_full_image_name(data_path,list_of_image_name):
    new_image_names = []
    for image in list_of_image_name:
        full_path = data_path + "/" + image + ".jpeg"
        new_image_names.append(full_path)
    return new_image_names

def remove_nonexistent_data(list_of_health_data,list_of_image_name):
    new_list_of_health_data = []
    new_list_of_images = []
    for i in range(len(list_of_image_name)):
        if os.path.isfile(list_of_image_name[i]):
            new_list_of_health_data.append(list_of_health_data[i])
            new_list_of_images.append(list_of_image_name[i])
    
    return new_list_of_health_data,new_list_of_images


def resize_image(image_path,width,height,output_dir):
    image_dir = output_dir + "/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for image in image_path:
        name = os.path.basename(image)
        #print(name)
        img = Image.open(image)
        new_img = img.resize((width,height))
        save_location = image_dir + "/" + name
        print("resized image to: ",save_location)
        new_img.save(save_location, "JPEG", optimize=True)

def normalize_images(list_of_image_name):
    list_of_normalized_images = []
    counter = 0
    for image in list_of_image_name:
        #print(image)
        normalized_image = cv2.imread(image)
        normalized_image = normalized_image/255.0
        list_of_normalized_images.append(normalized_image)
    return list_of_normalized_images

def split_data_train_test(list_of_health_data,list_of_image_name,percent_for_test):
    list_of_image_name_train = []
    list_of_image_name_test = []
    list_of_health_data_train = []
    list_of_health_data_test = []

    number_of_tot_data = len(list_of_image_name)
    number_of_test_data = number_of_tot_data*percent_for_test
    number_of_test_data = math.floor(number_of_test_data)

    list_of_image_name_train = list_of_image_name[number_of_test_data:]
    list_of_image_name_test = list_of_image_name[:number_of_test_data]
    list_of_health_data_train = list_of_health_data[number_of_test_data:]
    list_of_health_data_test = list_of_health_data[:number_of_test_data]

    # print(len(list_of_health_data_train))
    # print(len(list_of_image_name_train))
    # print(len(list_of_health_data_test))
    # print(len(list_of_image_name_test))

    get_info_on_data(list_of_health_data_train)
    get_info_on_data(list_of_health_data_test)
    return list_of_image_name_train, list_of_health_data_train, list_of_image_name_test, list_of_health_data_test

def save_model(model_to_save):
    current_dir = os.getcwd()
    output_dir = current_dir + "/checkpoints"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    number_of_checkpoints = os.listdir(output_dir)
    number_of_checkpoints = len(number_of_checkpoints)
    new_checkpoint_number = number_of_checkpoints + 1
    model_name = "checkpoint" + str(number_of_checkpoints)
    model_name = output_dir + "/" + model_name
    model_to_save.save(model_name)

def load_model(path_to_model):
    print("Loading Model...")
    model_to_load = models.load_model(path_to_model)
    return model_to_load



# def save_csv(csv_list,image_name,output_dir)
#     label_dir = output_dir + "/labels"
#     if not os.path.exists(label_dir):
#         os.makedirs(label_dir)
#     for i in range(len(label_list)):
#     data = pd.DataFrame(csv_list,)



if __name__ == "__main__":
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

    if image_dir is None:
        raise SyntaxError('directory for images must be provided')

    if csv_dir is None:
        raise SyntaxError('Location for data labels csv file must be provided')

    if run_mode == 1:
        if output_dir is None:
            raise SyntaxError('Must provide a directory for the output')
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        #there are 30ish missing files, should make a new csv later
        health_level,image_name = remove_nonexistent_data(health_level,image_name)
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



        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(new_image_width, new_image_height, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(layers.Dense(5, activation='softmax'))        
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        #model.summary()

        #loads a model if a flag is provided
        if model_to_load is not None:
            model = load_model(model_to_load)
        
        
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
                model.evaluate(np_image_batch_test, test_lables)
            #model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.Accuracy()])
            image_batch.clear()
            label_batch.clear()
            current_epoch = images_trained_on/total_images
            if current_epoch - save_interval>previous_save:
                save_model(model)
                previous_save = previous_save + save_interval
            print("epoch: ",current_epoch)
   