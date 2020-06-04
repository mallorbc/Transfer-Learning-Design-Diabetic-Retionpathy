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

import gc
from sklearn.model_selection import train_test_split


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
        
    save_csv(run_dir,"garbage.csv",discarded_health_data,discarded_image_names)
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
            normalized_image = cv2.cvtColor(normalized_image,cv2.COLOR_BGR2RGB)
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

def split_data_train_test_val(run_dir,list_of_health_data,list_of_image_name,percent_for_test,val_percent):
    list_of_image_name_train = []
    list_of_image_name_test = []
    list_of_health_data_train = []
    list_of_health_data_test = []
    list_of_image_name_val = []
    list_of_health_data_val = []
    
    list_of_image_name_train,list_of_image_name_val,list_of_health_data_train,list_of_health_data_val = train_test_split(list_of_image_name,list_of_health_data,test_size=val_percent,random_state=1)
    list_of_image_name_train,list_of_image_name_test,list_of_health_data_train,list_of_health_data_test = train_test_split(list_of_image_name_train,list_of_health_data_train,test_size=percent_for_test,random_state=1)

    utils.get_info_on_data(list_of_health_data_train)
    utils.get_info_on_data(list_of_health_data_test)
    utils.get_info_on_data(list_of_health_data_val)


    save_csv(run_dir,"train.csv",list_of_health_data_train,list_of_image_name_train)
    save_csv(run_dir,"test.csv",list_of_health_data_test,list_of_image_name_test)
    save_csv(run_dir,"validation.csv",list_of_health_data_val,list_of_image_name_val)

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

def regcrop(image_path,output_dir,save_numpy):
    """
    Basic crop_image_from_gray for a dataset
    """
    image_dir = output_dir + "/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for image in image_path:
        name = os.path.basename(image)
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        if img is None:
            continue

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #gets the height and width
        height, width, channels = img.shape
        image_size = height
        #adds a weight blur as done in the online kaggle demo
        img=cv2.addWeighted(img,4, cv2.GaussianBlur(img , (0,0) , 5) ,-4 ,128)
        #saves the modified image in the output directory
        save_location = image_dir + "/" + name
        img = Image.fromarray(img)
        img.save(save_location,"JPEG", optimize=True)
        counter = counter + 1
        print(counter)

def zca_nocirc(image_path,output_dir, batch_size, blur):
    """
    Crops (not circular)
    Performs ZCA whitening on images
    """
    image_dir = output_dir + "/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filepath_list = []
    imagecounter = 0

    for image in image_path:
        #print("Image path: "+str(image_path))
        print("Image value: "+str(image))
        filepath_list.append(image)
        print(os.path.basename(image) + " added to master list.")
        imagecounter = imagecounter + 1
    
    #print("Image path: "+str(image_path))
    print("Imagecounter value: " + str(imagecounter))
    print("Master filepath list made. Length: " + str(len(filepath_list)))
    #print(filepath_list)

    
    num_batch = int(len(filepath_list)/batch_size) + 1
    #num_batch = int(len(image_dir)/batch_size) + 1
    print("Number of batches: " + str(num_batch))

    container = []

    #for i in range(0, num_batch):
        #container.append([])
    
    for i in range(0, num_batch):
        temp = []
        for j in range(0, batch_size):
            filepath_pos = i*batch_size+j
            if filepath_pos < len(filepath_list):
                print("Adding file "+str(filepath_list[filepath_pos])+" to batch "+str(i+1))
                temp.append(filepath_list[filepath_pos])
        container.append(temp)

    #print("Count of last batch: "+str(len(container[8]))+" images.")
    print("Number of batches in container: "+str(len(container)))
    for i in range(0, num_batch):
        print(str(len(container[i])))
    print("Last image of batch 1: "+container[0][batch_size-1])
    #sys.exit("Exiting program.")

    for batch in container:         #Runs on a per-batch basis to avoid memory overflow
        count = 1
        X_names = []

        for image in batch:          #Reads, changes to RGB from BGR, crops, and resizes to 512x512.
            name = os.path.basename(image)
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = crop_image_from_gray(img)
            img = cv2.resize(img, (512, 512))
            if blur == True:
                img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)

            if count == 1:              #adds image numpy array to an array containing the data from 1 image in each row
                X = np.array([img])
                X_names.append(name)
                print("Image shape: ", img.shape)
                print("Processed image: " + name + "\n")
            else:
                X = np.append(X, [img], axis=0)
                X_names.append(name)
                print("Image shape: ", img.shape)
                print("Processed image: " + name + "\n")
            count = count + 1
            print(str(count))

        print("X.shape: ", X.shape)
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])  #flattening image data to 1 row
        print("Reshaped images.")

        X_norm = X/255          #size and mean normalizing on per-pixel basis
        mean = X_norm.mean(axis=0)
        X_norm = X_norm - mean
        print("Size and mean normalized images.")

        cov = np.cov(X_norm, rowvar = True)     #calculating covariance matrix, finding eigenvectors/values, defining epsilon
        U,S,V = np.linalg.svd(cov)
        epsilon = 0.001

        print("Performing whitening.")
        X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_norm)    #performing whitening
        
        print("Rescaling to normal RGB range.")
        X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min()) * 255    #rescaling back to normal image values

        count = 0
        os.chdir(image_dir)

        print("Creating image files..")
        for imgarray in X_ZCA_rescaled:
            saved_img = (X_ZCA_rescaled[count,:].reshape(512,512,3))[:,:,[2,1,0]]
            cv2.imwrite(X_names[count], saved_img)
            print("Image file " + X_names[count] + " created.")
            count = count + 1

        X = None
        X_norm = None
        cov = None
        X_ZCA = None
        X_ZCA_rescaled = None
        saved_img = None
        
        print("Function completed.")

def prepare_data_for_model_rand(size_of_data,labels,images,image_width,image_height):
    gc.collect()
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

def prepare_data_for_model(size_of_data,labels,images,image_width,image_height,health_dict,method=1):
    if method == 1:
        np_image_batch,labels_batch = prepare_data_for_model_rand(size_of_data,labels,images,image_width,image_height)
    elif method == 2:
        np_image_batch,labels_batch = prepare_data_for_model_even(size_of_data,labels,images,image_width,image_height,health_dict)

    return np_image_batch,labels_batch
    print("test")

def prepare_data_for_model_even(size_of_data,labels,images,image_width,image_height,health_dict):
    gc.collect()
    total_labels = []
    images,labels = generate_even_classes(images,labels,health_dict)
    labels,images = shuffle_data(labels,images)
    total_labels = labels
    if len(images)>size_of_data:
        image_batch = images[:size_of_data]
        labels_batch = total_labels[:size_of_data]
    else:
        image_batch = images
        labels_batch = total_labels
    # shuffle_data(labels_batch,image_batch)
    image_batch = normalize_images(image_batch)
    np_image_batch = np.asarray(image_batch)
    np_image_batch.reshape(len(image_batch),image_width,image_height,3)
    labels_batch = np.asarray(labels_batch)


    return np_image_batch,labels_batch

def prepare_data_for_model_even_binary(size_of_data,labels,images,image_width,image_height,health_dict):
    gc.collect()
    total_labels = []
    images,labels = generate_even_classes_binary(images,labels,health_dict)
    labels,images = shuffle_data(labels,images)
    total_labels = labels
    if len(images)>size_of_data:
        image_batch = images[:size_of_data]
        labels_batch = total_labels[:size_of_data]
    else:
        image_batch = images
        labels_batch = total_labels
    # shuffle_data(labels_batch,image_batch)
    image_batch = normalize_images(image_batch)
    np_image_batch = np.asarray(image_batch)
    np_image_batch.reshape(len(image_batch),image_width,image_height,3)
    labels_batch = np.asarray(labels_batch)


    return np_image_batch,labels_batch

def prepare_data_for_model_two(size_of_data,labels,images,second_images,image_width,image_height):
    gc.collect()
    total_labels = []
    # labels,images = shuffle_data(labels,images)
    labels,images,second_images = shuffle(labels,images,second_images)
    total_labels = labels
    image_batch = images[:size_of_data]
    image_batch_two = second_images[:size_of_data]
    labels_batch = total_labels[:size_of_data]
    np_labels_batch = np.asarray(labels_batch)
    image_batch = normalize_images(image_batch)
    image_batch_two = normalize_images(image_batch_two)
    np_image_batch = np.asarray(image_batch)
    np_image_batch_two = np.asarray(image_batch_two)
    np_image_batch.reshape(len(image_batch),image_width,image_height,3)
    np_image_batch.reshape(len(image_batch_two),image_width,image_height,3)


    return np_image_batch,np_image_batch_two,np_labels_batch

def save_data_as_np(list_of_images):
    print("save numpy")

def make_csv_dict(health_level,image_name):
    pickle_dict = {}

    for i in range(len(image_name)):
        # print(image_name[i])
        pickle_dict[os.path.basename(image_name[i])] = health_level[i]
        # print(str(i) + " total entries in dict")
    
    return pickle_dict



def cut_data_class(health_dict,class_to_cut,percent_to_cut):
    #these are the values that will be returned
    all_images = []
    all_images = np.asarray(all_images)
    all_labels = []
    all_labels = np.asarray(all_labels)
    #grabs the proper list to cut
    images_to_cut = health_dict[class_to_cut]
    #gets the number to slice the list by
    number_to_cut = int(len(images_to_cut)*percent_to_cut)
    #shuffles the list before cutting it
    images_to_cut = shuffle(images_to_cut)
    #cuts the list
    images_to_cut = images_to_cut[:number_to_cut]
    cut_labels_number = len(images_to_cut)
    #labels of the cut class to add
    cut_labels = np.ones(cut_labels_number)
    cut_labels = np.multiply(cut_labels,class_to_cut)
    # quit()
    all_labels = np.append(all_labels,cut_labels)
    #converts the list into a np array
    images_to_cut = np.asarray(images_to_cut)
    #creates a list of all images starting with the ones cut
    all_images = np.append(all_images,images_to_cut)
    #gets all the keys
    all_keys = list(health_dict.keys())
    #loops through all the keys
    for key in all_keys:
        #if the key is  equal to the class we already cut, we have nothing to do
        if key == class_to_cut:
            continue

        images_of_class = health_dict[key]
        number_of_images = len(images_of_class)
        labels = np.ones(number_of_images)
        labels = np.multiply(labels,key)
        all_images = np.append(all_images,images_of_class)
        all_labels = np.append(all_labels,labels)
    #converts labels to int
    all_labels = all_labels.astype(int)

    return all_labels,all_images

def generate_even_classes(images,labels,health_dict):
    cat0,cat1,cat2,cat3,cat4 = utils.get_info_on_data(labels)
    categories = [cat0,cat1,cat2,cat3,cat4]
    categories = np.asarray(categories)
    lowest_index = np.argmin(categories)
    lowest_value = categories[lowest_index]
    print(lowest_value)
    class_0 = health_dict[0]
    class_1 = health_dict[1]
    class_2 = health_dict[2]
    class_3 = health_dict[3]
    class_4 = health_dict[4]

    class_0 = shuffle(class_0)
    class_1 = shuffle(class_1)
    class_2 = shuffle(class_2)
    class_3 = shuffle(class_3)
    class_4 = shuffle(class_4)

    class_0 = class_0[:lowest_value]
    class_1 = class_1[:lowest_value]
    class_2 = class_2[:lowest_value]
    class_3 = class_3[:lowest_value]
    class_4 = class_4[:lowest_value]

    all_images = []
    all_images = np.asarray(all_images)
    all_labels = []
    all_labels = np.asarray(all_labels)

    all_images = np.append(all_images,class_0)
    all_labels = np.append(all_labels,np.multiply(np.ones(len(class_0)),0))

    all_images = np.append(all_images,class_1)
    all_labels = np.append(all_labels,np.multiply(np.ones(len(class_1)),1))

    all_images = np.append(all_images,class_2)
    all_labels = np.append(all_labels,np.multiply(np.ones(len(class_2)),2))

    all_images = np.append(all_images,class_3)
    all_labels = np.append(all_labels,np.multiply(np.ones(len(class_3)),3))

    all_images = np.append(all_images,class_4)
    all_labels = np.append(all_labels,np.multiply(np.ones(len(class_4)),4))

    # print(all_labels)
    # print(len(all_labels))
    # print(len(all_images))




    return all_images,all_labels 

def generate_even_classes_binary(images,labels,health_dict):
    cat0,cat1,cat2,cat3,cat4 = utils.get_info_on_data(labels)
    categories = [cat0,cat1]
    categories = np.asarray(categories)
    lowest_index = np.argmin(categories)
    lowest_value = categories[lowest_index]
    print(lowest_value)
    class_0 = health_dict[0]
    class_1 = health_dict[1]


    class_0 = shuffle(class_0)
    class_1 = shuffle(class_1)


    class_0 = class_0[:lowest_value]
    class_1 = class_1[:lowest_value]


    all_images = []
    all_images = np.asarray(all_images)
    all_labels = []
    all_labels = np.asarray(all_labels)

    all_images = np.append(all_images,class_0)
    all_labels = np.append(all_labels,np.multiply(np.ones(len(class_0)),0))

    all_images = np.append(all_images,class_1)
    all_labels = np.append(all_labels,np.multiply(np.ones(len(class_1)),1))


    return all_images,all_labels 


def generate_dataframe(images,labels,health_dict,method):
    if method == 1:
        # data_frame_data = [images,labels]
        # df = pd.DataFrame(data_frame_data,columns=["image","label"])
        df = pd.DataFrame({"image": images,
                            "label": labels})
    elif method == 2:
        images,labels = generate_even_classes(images,labels,health_dict)
        df = pd.DataFrame({"image": images,"label": labels})
    return df

def convert_labels_to_binary(labels):
    new_labels = []
    for label in labels:
        if label>0:
            new_labels.append(1)
        else:
            new_labels.append(0)
    return new_labels
