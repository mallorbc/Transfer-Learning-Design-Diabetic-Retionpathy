#used for viewing images
import matplotlib.pyplot as plt
#model stuff
import numpy as np
#for getting data paths
import os
#for image loading and manipulation
import cv2

def add_plot_data(accuracy_test,accuracy_train,epoch,run_dir):
    #creates the directory if it does not exist
    current_dir = run_dir
    output_dir = current_dir + "/plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    epochs_file_path = output_dir + "/" + "epochs.npy"
    accuracy_file_path_test = output_dir + "/" + "accuracy_test.npy"
    accuracy_file_path_train = output_dir + "/" + "accuracy_train.npy"
    #if the files exists we load it first
    if os.path.isfile(epochs_file_path) and os.path.isfile(accuracy_file_path_test) and os.path.isfile(accuracy_file_path_train):
        #loads the existing files
        epochs_numpy_file = np.load(epochs_file_path)
        accuracy_numpy_file_test = np.load(accuracy_file_path_test)
        accuracy_numpy_file_train = np.load(accuracy_file_path_train)
        #appends the new data to the array
        accuracy_numpy_file_test = np.append(accuracy_numpy_file_test,accuracy_test)
        accuracy_numpy_file_train = np.append(accuracy_numpy_file_train,accuracy_train)
        epochs_numpy_file = np.append(epochs_numpy_file,epoch)
        #saves the arrays
        np.save(epochs_file_path,epochs_numpy_file)
        np.save(accuracy_file_path_test,accuracy_numpy_file_test)
        np.save(accuracy_file_path_train,accuracy_numpy_file_train)
        print("Added data to numpy file")
    #if the file does not exist we create it
    else:
        epochs_numpy_file = []
        accuracy_numpy_file_test = []
        accuracy_numpy_file_train = []
        #adds the data to the lists
        accuracy_numpy_file_test = np.append(accuracy_numpy_file_test,accuracy_test)
        accuracy_numpy_file_train = np.append(accuracy_numpy_file_train,accuracy_train)
        epochs_numpy_file = np.append(epochs_numpy_file,epoch)
        #converts these lists to numpy arrays
        epochs_numpy_file = np.asarray(epochs_numpy_file)
        accuracy_numpy_file_test = np.asarray(accuracy_numpy_file_test)
        accuracy_numpy_file_train = np.asarray(accuracy_numpy_file_train)
        #saves the arrays
        np.save(epochs_file_path,epochs_numpy_file)
        np.save(accuracy_file_path_test,accuracy_numpy_file_test)
        np.save(accuracy_file_path_train,accuracy_numpy_file_train)


def plot_accuracy(data_directory):
    #used to build error plot
    error_array_test = []
    error_array_train = []
    #builds file paths to load files
    current_dir = data_directory
    plot_dir = current_dir + "/" + "plots"
    epochs_file = plot_dir + "/" + "epochs.npy"
    accuracy_file_test = plot_dir + "/" + "accuracy_test.npy"
    accuracy_file_train = plot_dir + "/" + "accuracy_train.npy"
    #loads the arrays
    epochs_array = np.load(epochs_file)
    accuracy_array_test = np.load(accuracy_file_test)
    accuracy_array_train = np.load(accuracy_file_train)
    #calculates and saves the error
    for accuracy in accuracy_array_test:
        error = 1 - accuracy
        error_array_test.append(error)
    for accuracy in accuracy_array_train:
        error = 1 - accuracy
        error_array_train.append(error)
    #plots and formats accuracy and error for the test data
    plt.plot(epochs_array,accuracy_array_test,label="Accuracy Test")
    #plt.plot(epochs_array,error_array_test,label="Error Test")
    #plots and formats the accuracy and error for the train data
    plt.plot(epochs_array,accuracy_array_train,label="Accuracy Train")
    #plt.plot(epochs_array,error_array_train,label="Error Train")
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