#easy resizing images
from PIL import Image
#for getting data paths
import os
import numpy as np

import preprocessData
import time
import argparse

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

def get_full_image_name_no_ext(data_path,list_of_image_name):
    new_image_names = []
    for image in list_of_image_name:
        full_path = data_path + "/" + image
        new_image_names.append(full_path)
    return new_image_names

def get_previous_directory(current_dir):
    previous_dir = os.path.abspath(os.path.join(os.path.dirname(current_dir ), '..'))
    return previous_dir

def get_two_directories_up(current_dir):
    previous_dir = os.path.abspath(os.path.join(os.path.dirname(current_dir ), '../..'))
    return previous_dir

def get_last_epoch(epoch_file):
    epoch_array = np.load(epoch_file)
    last_epoch = epoch_array[-1]
    return last_epoch

def get_lowest_loss(epoch_file,loss_file):
    epoch_array = np.load(epoch_file)
    loss_array = np.load(loss_file)
    low_loss_index = np.where(loss_array == np.amin(loss_array))
    lowest_loss_epoch = epoch_array[low_loss_index]
    lowest_loss_value = loss_array[low_loss_index]

    return lowest_loss_value,lowest_loss_epoch

def get_highest_accuracy(epoch_file,acc_file):
    epoch_array = np.load(epoch_file)
    accuracy_array = np.load(acc_file)
    highest_acc_index = np.where(accuracy_array == np.amax(accuracy_array))
    highest_acc_epoch = epoch_array[highest_acc_index]
    highest_acc_value = accuracy_array[highest_acc_index]
    return highest_acc_value,highest_acc_epoch

def get_trimmed_data(number_to_get,trimmed_images,trimmed_labels):
    trimmed_labels,trimmed_images = preprocessData.shuffle_data(trimmed_labels,trimmed_images)
    trimmed_labels = trimmed_labels[:number_to_get]
    trimmed_images = trimmed_images[:number_to_get]
    return trimmed_images,trimmed_labels


def change_dir_name(new_dir,images):
    return_list = []
    for image in images:
        temp = os.path.basename(image)
        temp = new_dir + temp
        return_list.append(os.path.realpath(temp))
    return return_list


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_extension(list_of_files,extension_to_add):
    new_list = []
    for file in list_of_files:
        new_list.append(file + extension_to_add)
    return new_list

def get_images_of_one_class(class_to_get,list_of_images,class_dict):
    images_of_class = []

    for image in list_of_images:
        if class_dict[os.path.basename(image)]!=class_to_get:
            continue
        else:
            images_of_class.append(image)
    
    return images_of_class

def make_npy_of_class(class_to_get,list_of_images,class_dict,output_folder,csv_file=None):
    images_of_one_class = get_images_of_one_class(class_to_get,list_of_images,class_dict)
    images_of_one_class = add_extension(images_of_one_class,".jpeg")
    images_to_use = []

    if csv_file is not None:
        class_name,images_from_csv = preprocessData.load_data(csv_file)
        for i in range(len(images_of_one_class)):
            for j in range(len(images_from_csv)):
                # print(images_from_csv[j])
                # print(images_of_one_class[i])
                # quit()
                if os.path.basename(images_from_csv[j]) == os.path.basename(images_of_one_class[i]):
                    images_to_use.append(images_of_one_class[i])
                    break
    else:
        images_to_use = images_of_one_class
    
    print(len(images_to_use))
    # quit()

    save_location_base = os.path.realpath(output_folder)
    counter = 0
    for image in images_to_use:
            image_name = image
            image = Image.open(image)
            pixels=np.asarray(image)
            pixels = pixels.astype('float32')
            pixels /=255.0
            print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
            save_location = save_location_base + "/" + os.path.basename(image_name)
            np.save(save_location,pixels)
            counter = counter + 1
            print("Made "+str(counter)+" npy files")

