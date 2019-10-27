#used for viewing images
import matplotlib.pyplot as plt
#model stuff
import numpy as np
#for getting data paths
import os
#for image loading and manipulation
import cv2

def add_plot_data(accuracy,epoch,date_time):
    #creates the directory if it does not exist
    current_dir = os.getcwd()
    output_dir = current_dir + "/" + date_time + "/" + "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    epochs_file_path = output_dir + "/" + "epochs.npy"
    accuracy_file_path = output_dir + "/" + "accuracy.npy"
    #if the files exists we load it first
    if os.path.isfile(epochs_file_path) and os.path.isfile(accuracy_file_path):
        #loads the existing files
        epochs_numpy_file = np.load(epochs_file_path)
        accuracy_numpy_file = np.load(accuracy_file_path)
        #appends the new data to the array
        accuracy_numpy_file = np.append(accuracy_numpy_file,accuracy)
        epochs_numpy_file = np.append(epochs_numpy_file,epoch)
        #saves the arrays
        np.save(epochs_file_path,epochs_numpy_file)
        np.save(accuracy_file_path,accuracy_numpy_file)
        print("Added data to numpy file")
    #if the file does not exist we create it
    else:
        epochs_numpy_file = []
        accuracy_numpy_file = []
        #adds the data to the lists
        accuracy_numpy_file = np.append(accuracy_numpy_file,accuracy)
        epochs_numpy_file = np.append(epochs_numpy_file,epoch)
        #converts these lists to numpy arrays
        epochs_numpy_file = np.asarray(epochs_numpy_file)
        accuracy_numpy_file = np.asarray(accuracy_numpy_file)
        #saves the arrays
        np.save(epochs_file_path,epochs_numpy_file)
        np.save(accuracy_file_path,accuracy_numpy_file)

def plot_accuracy():
    #used to build error plot
    error_array = []
    #builds file paths to load files
    current_dir = os.getcwd()
    plot_dir = current_dir + "/" + "plots"
    epochs_file = plot_dir + "/" + "epochs.npy"
    accuracy_file = plot_dir + "/" + "accuracy.npy"
    #loads the arrays
    epochs_array = np.load(epochs_file)
    accuracy_array = np.load(accuracy_file)
    #calculates and saves the error
    for accuracy in accuracy_array:
        error = 1 - accuracy
        error_array.append(error)
    #plots and formats accuracy and error
    plt.plot(epochs_array,accuracy_array,label="Accuracy")
    plt.plot(epochs_array,error_array,label="Error")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/Error")
    plt.title("Accuracy/Error vs Epoch")
    plt.legend()
    plt.show()


def show_images(image_name):
    #loops through all images
    for image in image_name:
        img = cv2.imread(image)
        #im_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()