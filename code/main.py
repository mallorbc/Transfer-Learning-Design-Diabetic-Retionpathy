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
from PIL import Image




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

def shuffle_data():
    #shuffles the data together
    health_level,image_name = shuffle(health_level,image_name)


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
    cat0_counter = 0
    for i in range(len(list_of_health_data)):
        if list_of_health_data[i] !=0:
            new_list_of_health_data.append(list_of_health_data[i])
            new_list_of_images.append(list_of_image_name[i])
            cat0_counter = cat0_counter +1

        elif list_of_health_data[i] == 0 and cat0_counter %5 == 0:
            new_list_of_health_data.append(list_of_health_data[i])
            new_list_of_images.append(list_of_image_name[i])
    return new_list_of_health_data,new_list_of_images
    

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
        print(save_location)
        new_img.save(save_location, "JPEG", optimize=True)

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
    args = parser.parse_args()
    image_dir = args.dir
    csv_dir = args.csv_location
    run_mode = args.mode
    output_dir = args.output_dir
    new_image_width = args.image_width
    new_image_height = args.image_height
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

    #gets the path of the data
    data_path = os.path.abspath(image_dir)
    
    health_level,image_name = load_data(csv_dir)
    image_name = get_full_image_name(data_path,image_name)
    #health_level,image_name = remove_nonexistent_data(health_level,image_name)

    #show_images(image_name)

    #this shows that the data has way to many zeros
    #get_info_on_data(health_level)
    #health_level,image_name = trim_data(health_level,image_name)
    #get_info_on_data(health_level)
    # print(len(health_level))
    # print(len(image_name))
    #get_image_width_height(image_name)
    resize_image(image_name,new_image_width,new_image_height,output_dir)

