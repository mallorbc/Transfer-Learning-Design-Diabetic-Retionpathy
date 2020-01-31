#used to read in the csv file
import pandas as pd
#for image loading and manipulation
import cv2
#used to shuffle the data
from sklearn.utils import shuffle
#easy resizing images
from PIL import Image
import os

import matplotlib.pyplot as plt

import time

import glob

import math

import numpy as np

import utils


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


#this cuts down on the number of zeros
def trim_data(run_dir,list_of_health_data,list_of_image_name):
    cat0,cat1,cat2,cat3,cat4 = utils.get_info_on_data(list_of_health_data)
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
    
    save_csv(run_dir,"trimmed.csv",discarded_health_data,discarded_image_names)

    return new_list_of_health_data,new_list_of_images

    # return new_list_of_health_data,new_list_of_images,discarded_image_names,discarded_health_data

def trim_data_even(run_dir,list_of_health_data,list_of_image_name,size_of_each_class):
    cat0,cat1,cat2,cat3,cat4 = utils.get_info_on_data(list_of_health_data)
    # print(cat0)
    # print(cat1)
    # print(cat2)
    # print(cat4)
    new_list_of_health_data = []
    new_list_of_images = []
    discarded_image_names = []
    discarded_health_data = []

    #gets the modulys for each class
    cat0_mod = int(cat0/size_of_each_class)
    cat1_mod = int(cat1/size_of_each_class)
    cat2_mod = int(cat2/size_of_each_class)
    cat3_mod = int(cat3/size_of_each_class)
    cat4_mod = int(cat4/size_of_each_class)
    #counter that will be used to increase the modulo
    cat0_counter = 0
    cat1_counter = 0
    cat2_counter = 0
    cat3_counter = 0
    cat4_counter = 0

    total_0 = 0
    total_1 = 0
    total_2 = 0
    total_3 = 0
    total_4 = 0

    for i in range(len(list_of_image_name)):
        current_image_class = list_of_health_data[i]
        if current_image_class == 0 and total_0<size_of_each_class:
            cat0_counter = cat0_counter + 1
            if cat0_counter%cat0_mod == 0:
                new_list_of_health_data.append(list_of_health_data[i])
                new_list_of_images.append(list_of_image_name[i])
                cat0_counter = 0
                total_0 = total_0 + 1

            else:
                discarded_image_names.append(list_of_image_name[i])
                discarded_health_data.append(list_of_health_data[i])

        elif current_image_class == 1 and total_1<size_of_each_class:
            cat1_counter = cat1_counter + 1
            if cat1_counter%cat1_mod == 0:
                new_list_of_health_data.append(list_of_health_data[i])
                new_list_of_images.append(list_of_image_name[i])
                cat1_counter = 0
                total_1 = total_1 + 1
            else:
                discarded_image_names.append(list_of_image_name[i])
                discarded_health_data.append(list_of_health_data[i])

        elif current_image_class == 2 and total_2<size_of_each_class:
            cat2_counter = cat2_counter + 1
            if cat2_counter%cat2_mod == 0:
                new_list_of_health_data.append(list_of_health_data[i])
                new_list_of_images.append(list_of_image_name[i])
                cat2_counter = 0
                total_2 = total_2 + 1

            else:
                discarded_image_names.append(list_of_image_name[i])
                discarded_health_data.append(list_of_health_data[i])

        elif current_image_class == 3 and total_3<size_of_each_class:
            cat3_counter = cat3_counter + 1
            if cat3_counter%cat3_mod == 0:
                new_list_of_health_data.append(list_of_health_data[i])
                new_list_of_images.append(list_of_image_name[i])
                cat3_counter = 0
                total_3 = total_3 + 1

            else:
                discarded_image_names.append(list_of_image_name[i])
                discarded_health_data.append(list_of_health_data[i])

        elif current_image_class == 4:
            cat4_counter = cat4_counter + 1
            if cat4_counter%cat4_mod == 0 and total_4<size_of_each_class:
                new_list_of_health_data.append(list_of_health_data[i])
                new_list_of_images.append(list_of_image_name[i])
                cat4_counter = 0
                total_4 = total_4 + 1

            else:
                discarded_image_names.append(list_of_image_name[i])
                discarded_health_data.append(list_of_health_data[i])
        
    save_csv(run_dir,"validation.csv",discarded_health_data,discarded_image_names)
    cat0,cat1,cat2,cat3,cat4 = utils.get_info_on_data(new_list_of_health_data)

    return new_list_of_health_data,new_list_of_images




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


def normalize_images(list_of_image_name,save_numpy=False):
    list_of_normalized_images = []
    counter = 0
    for image in list_of_image_name:
        #print(image)
        #print(counter)
        if save_numpy:
            normalized_image = image
        else:
            normalized_image = cv2.imread(image)
        normalized_image = normalized_image/255.0
        list_of_normalized_images.append(normalized_image)
        counter = counter + 1
    return list_of_normalized_images


def split_data_train_test(run_dir,list_of_health_data,list_of_image_name,percent_for_test):
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

    utils.get_info_on_data(list_of_health_data_train)
    utils.get_info_on_data(list_of_health_data_test)

    save_csv(run_dir,"train.csv",list_of_health_data_train,list_of_image_name_train)
    save_csv(run_dir,"test.csv",list_of_health_data_test,list_of_image_name_test)


    return list_of_image_name_train, list_of_health_data_train, list_of_image_name_test, list_of_health_data_test

def save_csv(run_dir,csv_name,health_data,image_name):
    csv_dir  = run_dir + "/csv_files"
    image = []
    #gets just the image name
    for item in image_name:
        image.append(os.path.basename(item))
    level = health_data
    #pust the data into a dataframe
    data = pd.DataFrame({"image" : image,"level" : level})
    #makes the directory if it does not exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    #creates the file and saves it
    csv_file_path = csv_dir + "/" + csv_name
    data.to_csv(csv_file_path)



def crop_image_from_gray(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    if img is None:
        return img
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    
    img = cv2.imread(img)
    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    
    return img 

def circle_crop_v2(image_path,output_dir,save_numpy):
    """
    Create circular crop around image centre
    """
    image_dir = output_dir + "/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for image in image_path:
        name = os.path.basename(image)
        img = cv2.imread(image)
        img = crop_image_from_gray(img)
        if img is None:
            continue

        height, width, depth = img.shape
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape

        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = crop_image_from_gray(img)
        save_location = image_dir + "/" + name
        if not save_numpy:
            img = Image.fromarray(img)
            img.save(save_location,"JPEG", optimize=True)
        else:
            img_np = normalize_images(img,save_numpy)
            save_location = save_location.split(".")
            save_location = save_location[0]
            save_location = save_location + ".npy"
            np.save(save_location,img_np)

def add_blur(image_path,output_dir):
    #creates the output for the blurred images
    image_dir = output_dir + "/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    counter = 0
    for image in image_path:
        #stores the name of the image to be saved
        name = os.path.basename(image)
        #reads in the image
        img = cv2.imread(image)
        #gets the height and width
        height, width, channels = img.shape
        image_size = height
        #adds a weight blur as done in the online kaggle demo
        img=cv2.addWeighted(img,4, cv2.GaussianBlur(img , (0,0) , 10) ,-4 ,128)
        #saves the modified image in the output directory
        save_location = image_dir + "/" + name
        img = Image.fromarray(img)
        img.save(save_location,"JPEG", optimize=True)
        counter = counter + 1
        print(counter)


def prepare_data_for_model(size_of_data,labels,images,image_width,image_height):
    total_labels = []
    labels,images = shuffle_data(labels,images)
    total_labels = labels
    image_batch = images[:size_of_data]
    labels_batch = total_labels[:size_of_data]
    image_batch = normalize_images(image_batch)
    np_image_batch = np.asarray(image_batch)
    np_image_batch.reshape(len(image_batch),image_width,image_height,3)
    labels_batch = np.asarray(labels_batch)


    return np_image_batch,labels_batch

def prepare_data_for_model_two(size_of_data,labels,images,second_images,image_width,image_height):
    total_labels = []
    # labels,images = shuffle_data(labels,images)
    labels,images,second_images = shuffle(labels,images,second_images)
    total_labels = labels
    image_batch = images[:size_of_data]
    image_batch_two = second_images[:size_of_data]
    labels_batch = total_labels[:size_of_data]
    image_batch = normalize_images(image_batch)
    image_batch_two = normalize_images(image_batch_two)
    np_image_batch = np.asarray(image_batch)
    np_image_batch_two = np.asarray(image_batch_two)
    np_image_batch.reshape(len(image_batch),image_width,image_height,3)
    np_image_batch.reshape(len(image_batch_two),image_width,image_height,3)


    return np_image_batch,np_image_batch_two,labels_batch

def save_data_as_np(list_of_images):
    print("save numpy")

def make_csv_dict(health_level,image_name):
    pickle_dict = {}

    for i in range(len(image_name)):
        # print(image_name[i])
        pickle_dict[os.path.basename(image_name[i])] = health_level[i]
        # print(str(i) + " total entries in dict")
    
    return pickle_dict

#This will create a dict of classification numbers, with the stored values being a list of images of that class
def make_diagnose_class_dict(health_level,image_name):
    print("test")
# def get_data_for_prediction(images,labels):
