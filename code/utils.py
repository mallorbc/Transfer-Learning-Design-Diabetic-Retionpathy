#easy resizing images
from PIL import Image
#for getting data paths
import os

import numpy as np

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

def get_last_epoch(epoch_file):
    epoch_array = np.load(epoch_file)
    last_epoch = epoch_array[-1]
    return last_epoch