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


def show_images(folder,image_name):
    #loops through all images
    for image in image_name:
        #builds the name of the image path
        image_to_read = data_path +"/" + image + ".jpeg"
        # img = mpimg.imread(image_to_read)
        img = cv2.imread(image_to_read)
        #im_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == "__main__":
    print("main") 
    #gets the path of the data
    data_path = os.path.abspath("../data/resized_train_cropped")
    health_level,image_name = load_data("../data/trainLabels_cropped.csv")
    show_images(data_path,image_name)

