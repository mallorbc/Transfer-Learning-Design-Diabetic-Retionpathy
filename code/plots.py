#used for viewing images
import matplotlib.pyplot as plt
#model stuff
import numpy as np
#for getting data paths
import os
#for image loading and manipulation
import cv2
#for confusion matrix
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
import seaborn as sn
import pickle


import math
import time

from preprocessData import *
from myModels import *
from utils import *

def add_plot_data(accuracy_test,accuracy_train,loss_test,loss_train,epoch,run_dir):
    #creates the directory if it does not exist
    current_dir = run_dir
    output_dir = current_dir + "/plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    epochs_file_path = output_dir + "/" + "epochs.npy"
    accuracy_file_path_test = output_dir + "/" + "accuracy_test.npy"
    accuracy_file_path_train = output_dir + "/" + "accuracy_train.npy"
    loss_file_path_test = output_dir + "/" + "loss_test.npy"
    loss_file_path_train = output_dir + "/" + "loss_train.npy"
    #if the files exists we load it first
    if os.path.isfile(epochs_file_path) and os.path.isfile(accuracy_file_path_test) and os.path.isfile(accuracy_file_path_train) and os.path.isfile(loss_file_path_test) and os.path.isfile(loss_file_path_train):
        #loads the existing files
        epochs_numpy_file = np.load(epochs_file_path)
        accuracy_numpy_file_test = np.load(accuracy_file_path_test)
        accuracy_numpy_file_train = np.load(accuracy_file_path_train)
        loss_numpy_file_test = np.load(loss_file_path_test)
        loss_numpy_file_train = np.load(loss_file_path_train)
        #appends the new data to the array
        accuracy_numpy_file_test = np.append(accuracy_numpy_file_test,accuracy_test)
        accuracy_numpy_file_train = np.append(accuracy_numpy_file_train,accuracy_train)
        loss_numpy_file_test = np.append(loss_numpy_file_test,loss_test)
        loss_numpy_file_train = np.append(loss_numpy_file_train,loss_train)
        epochs_numpy_file = np.append(epochs_numpy_file,epoch)
        #saves the arrays
        np.save(epochs_file_path,epochs_numpy_file)
        np.save(accuracy_file_path_test,accuracy_numpy_file_test)
        np.save(accuracy_file_path_train,accuracy_numpy_file_train)
        np.save(loss_file_path_test,loss_numpy_file_test)
        np.save(loss_file_path_train,loss_numpy_file_train)
        print("Added data to numpy file")
    #if the file does not exist we create it
    else:
        epochs_numpy_file = []
        accuracy_numpy_file_test = []
        accuracy_numpy_file_train = []
        loss_numpy_file_test = []
        loss_numpy_file_train = []
        #adds the data to the lists
        accuracy_numpy_file_test = np.append(accuracy_numpy_file_test,accuracy_test)
        accuracy_numpy_file_train = np.append(accuracy_numpy_file_train,accuracy_train)
        loss_numpy_file_test = np.append(loss_numpy_file_test,loss_test)
        loss_numpy_file_train = np.append(loss_numpy_file_train,loss_train)
        epochs_numpy_file = np.append(epochs_numpy_file,epoch)
        #converts these lists to numpy arrays
        epochs_numpy_file = np.asarray(epochs_numpy_file)
        accuracy_numpy_file_test = np.asarray(accuracy_numpy_file_test)
        accuracy_numpy_file_train = np.asarray(accuracy_numpy_file_train)
        loss_numpy_file_test = np.asarray(loss_numpy_file_test)
        loss_numpy_file_train = np.asarray(loss_numpy_file_train)
        #saves the arrays
        np.save(epochs_file_path,epochs_numpy_file)
        np.save(accuracy_file_path_test,accuracy_numpy_file_test)
        np.save(accuracy_file_path_train,accuracy_numpy_file_train)
        np.save(loss_file_path_test,loss_numpy_file_test)
        np.save(loss_file_path_train,loss_numpy_file_train)


def plot_accuracy(data_directory,stop_epoch=None):
    #allows one to shorten the plots easily
    early_stop = -1
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
    if stop_epoch is not None:
        early_stop = np.where(epochs_array > stop_epoch)
        early_stop = early_stop[0][0]
    if early_stop!=-1:
        epochs_array = epochs_array[0:early_stop]
        accuracy_array_test = accuracy_array_test[0:early_stop]
        accuracy_array_train = accuracy_array_train[0:early_stop]

    #finds the highest accuracy and the epoch for both test and train
    high_accuracy_test_index = np.where(accuracy_array_test == np.amax(accuracy_array_test))
    high_accuracy_train_index = np.where(accuracy_array_train == np.amax(accuracy_array_train))
    high_accuracy_test = accuracy_array_test[high_accuracy_test_index]
    high_accuracy_train = accuracy_array_train[high_accuracy_train_index]
    high_accuracy_test_epoch = epochs_array[high_accuracy_test_index]
    high_accuracy_train_epoch = epochs_array[high_accuracy_train_index]
    #prints out the stats of the low loss values
    print("Highest test accuracy:",high_accuracy_test," at epoch:",high_accuracy_test_epoch)
    print("Highest train accuracy:",high_accuracy_train," at epoch:",high_accuracy_train_epoch)

    #calculates and saves the error
    for accuracy in accuracy_array_test:
        error = 1 - accuracy
        error_array_test.append(error)
    for accuracy in accuracy_array_train:
        error = 1 - accuracy
        error_array_train.append(error)
    #plots and formats accuracy and error for the test data
    plt.plot(epochs_array,accuracy_array_test,label="Accuracy Test")
    #plots and formats the accuracy and error for the train data
    plt.plot(epochs_array,accuracy_array_train,label="Accuracy Train")
    #plots the trend of test
    z = np.polyfit(epochs_array, accuracy_array_test, 1)
    p = np.poly1d(z)
    plt.plot(epochs_array,p(epochs_array),"r--",label="Test Trend")
    #plots the trend of the train
    z = np.polyfit(epochs_array, accuracy_array_train, 1)
    p = np.poly1d(z)
    plt.plot(epochs_array,p(epochs_array),"g--",label="Train Trend")

    #plt.plot(epochs_array,error_array_train,label="Error Train")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/Error")
    plt.title("Accuracy/Error vs Epoch")
    plt.legend()
    plt.show()

def plot_loss(data_directory,stop_epoch=None):
    #allows one to easily shorten the plots
    early_stop = -1
    #builds file paths to load files
    current_dir = data_directory
    plot_dir = current_dir + "/" + "plots"
    epochs_file = plot_dir + "/" + "epochs.npy"
    loss_file_test = plot_dir + "/" + "loss_test.npy"
    loss_file_train = plot_dir + "/" + "loss_train.npy"
    #loads the arrays
    epochs_array = np.load(epochs_file)
    loss_array_test = np.load(loss_file_test)
    loss_array_train = np.load(loss_file_train)
    if stop_epoch is not None:
        early_stop = np.where(epochs_array > stop_epoch)
        early_stop = early_stop[0][0]

    #shortens the arrays if desired
    if early_stop!=-1:
        loss_array_test = loss_array_test[0:early_stop]
        loss_array_train = loss_array_train[0:early_stop]
        epochs_array = epochs_array[0:early_stop]
    #gets the epochs and values of the lowest loss values
    low_loss_test_index = np.where(loss_array_test == np.amin(loss_array_test))
    low_loss_train_index = np.where(loss_array_train == np.amin(loss_array_train))
    low_loss_test = loss_array_test[low_loss_test_index]
    low_loss_train = loss_array_train[low_loss_train_index]
    low_loss_test_epoch = epochs_array[low_loss_test_index]
    low_loss_train_epoch = epochs_array[low_loss_train_index]
    #prints out the stats of the low loss values
    print("Lowest test loss:",low_loss_test," at epoch:",low_loss_test_epoch)
    print("Lowest train loss:",low_loss_train," at epoch:",low_loss_train_epoch)
    #plots and formats the loss for the test data
    plt.plot(epochs_array,loss_array_test,label="Loss Test")
    #plots and formats the loss for the train data
    plt.plot(epochs_array,loss_array_train,label="Loss Train")
    #plots the trend of test
    z = np.polyfit(epochs_array, loss_array_test, 1)
    p = np.poly1d(z)
    plt.plot(epochs_array,p(epochs_array),"r--",label="Test Trend")
    #plots the trend of the train
    z = np.polyfit(epochs_array, loss_array_train, 1)
    p = np.poly1d(z)
    plt.plot(epochs_array,p(epochs_array),"g--",label="Train Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.show()


def show_images(image_name):
    #loops through all images
    for image in image_name:
        img = cv2.imread(image)
        #im_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

def create_confusion_matrix_one_input(loaded_model,images,labels,plot_name=None,output=None):
    total_labels = []

    total_predictions = get_model_predictions_one_input(loaded_model,images)
    labels = labels[:len(total_predictions)]
    total_labels = np.asarray(labels)


    print(np.shape(total_predictions))
    print(np.shape(total_labels))

    print("Converting into matrix...")
    matrix = make_confusion_matrix_array(total_labels,total_predictions)
    report = classification_report(total_labels, total_predictions)
    print(report)
    if output is not None:
        dir_path = os.path.dirname(os.path.realpath(output))
        report_save_loc = dir_path + "/confusion_matrix.p"
        pickle.dump( report, open( report_save_loc, "wb" ) )
    print("Plotting...")
    plot_confusion_matrix(matrix,plot_name,output)

def create_confusion_matrix_two_inputs(model,images_one,images_two,labels):
    total_labels = []
    total_labels = np.asarray(labels)
    total_predictions = get_model_predictions_two_inputs(model,images_one,images_two)

    print(np.shape(total_predictions))
    print(np.shape(total_labels))

    print("Converting into matrix...")
    matrix = make_confusion_matrix_array(total_labels,total_predictions)
    report = classification_report(total_labels, total_predictions)
    print(report)
    # report_save_location =     
    # pickle.dump( report, open( "save.p", "wb" ) )
    print("Plotting...")
    plot_confusion_matrix(matrix,"test")


def make_confusion_matrix_array(actual, predicted):
    matrix = confusion_matrix(actual, predicted)
    return matrix




def plot_confusion_matrix(matrix, title=None,output=None):
    print("matrix")
    if title is None:
        title="confusion_matrix"

    array = matrix

    df_cm = pd.DataFrame(array, index=[i for i in "01234"], columns=[
                         i for i in "01234"])

    sn.set(font_scale=1.4)  # for label size
    ax = sn.heatmap(df_cm, annot=True, fmt='g',
                    annot_kws={"size": 20})  # font size
    ax.set(xlabel='Predicted', ylabel='Actual')
    plt.title(title)
    if output is not None:
        # save_name = output + "/"+ "confusion_matrix.png"
        plt.savefig(output)
        plt.close("all")

        # plt.show()

    else:
        plt.show()


#this will conver the data from 5 classes to a boolean, either the class or not and then also the confidence
def get_prob_of_correct(model,pickle_dict,list_of_images):
    return_values = []
    return_probs = []
    counter = 1
    for image in list_of_images:
        #gets the correct class
        correct_class_index = int(pickle_dict[os.path.basename(image)])
        #converts to a list to prepare to be normalized
        image = [image]
        image_to_test = normalize_images(image)
        image_to_test = np.asarray(image_to_test)
        #predicts and finds the class
        prediction_array = model.predict(image_to_test)
        prediction_array = np.squeeze(prediction_array,axis=0)
        # prediction_class = model.predict_classes(image_to_test)
        prediction_class = np.where(prediction_array == np.amax(prediction_array))
        # prediction_class = int(prediction_class)
        prediction_class = np.asarray(prediction_class)
        prediction_class = np.squeeze(prediction_class,axis=0)
        #sets the probs and results accordingly
        if prediction_class == correct_class_index:
            prediction_prob = float(prediction_array[correct_class_index])
            prediction_value = 1
        else:
            # print(prediction_array)
            # time.sleep(1)
            prediction_prob = float(1 - prediction_array[prediction_class])
            prediction_value = 0
        return_values.append(prediction_value)
        return_probs.append(prediction_prob)
        if counter >=12500:
            break
        print("Predicted "+str(counter) + " images")
        counter = counter + 1 
    return return_values,return_probs

def make_precision_recall_curve(class_to_test,images_to_test,class_dict,model_to_use):
    images_to_test = get_images_of_one_class(class_to_test,images_to_test,class_dict)
    results,probs = get_prob_of_correct(model_to_use,class_dict,images_to_test)
    plot_precision_recall_curve(results,probs)


def plot_precision_recall_curve(results,probs,output=None):
    precision, recall, thresholds = precision_recall_curve(results, probs)
    plt.plot(recall,precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    if output is not None:
        # save_path = output + "/precision_recall_curve.png"
        plt.savefig(output)
        # plt.show()
        plt.close("all")


    else:
        plt.show()



def make_roc_curve(class_to_test,images_to_test,class_dict,model_to_use):
    images_to_test = get_images_of_one_class(class_to_test,images_to_test,class_dict)
    results,probs = get_prob_of_correct(model_to_use,class_dict,images_to_test)
    plot_roc_curve(results,probs)

def plot_roc_curve(results,probs,output=None):
    fpr, tpr, thresholds = roc_curve(results, probs)
    try:
        score = roc_auc_score(results,probs)
        print("ROC AUC score: ",score)

    except:
        print("Can't give ROC score(its really bad)")
        time.sleep(3)

    base_line_x = np.arange(0.0,1.0,0.01)
    base_line_y = np.arange(0.0,1.0,0.01)
    plt.plot(fpr,tpr,label="ROC")
    plt.plot(base_line_x,base_line_y,"r--",label="Base Line")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if output is not None:
        # save_path = output + "/ROC.png"
        plt.savefig(output)
        plt.close("all")
        # plt.show()

    else:
        plt.show()


def make_roc_precision_recall_graphs(class_to_test,images_to_test,class_dict,model_to_use,output=None):
    images_to_test = get_images_of_one_class(class_to_test,images_to_test,class_dict)
    # print(images_to_test)
    # quit()
    results,probs = get_prob_of_correct(model_to_use,class_dict,images_to_test)
    if output is not None:
        output_file_precision = output + "/precision_recall_class_" + str(class_to_test) + ".png"
    plot_precision_recall_curve(results,probs,output_file_precision)
    output_file_roc = output + "/ROC_class_" + str(class_to_test) + ".png"
    plot_roc_curve(results,probs,output_file_roc)

def add_class_loss_data(loss0,loss1,loss2,loss3,loss4,run_dir):
    #creates the directory if it does not exist
    current_dir = run_dir
    output_dir = current_dir + "/plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    loss0_file_path = output_dir + "/" + "loss0.npy"
    loss1_file_path = output_dir + "/" + "loss1.npy"
    loss2_file_path = output_dir + "/" + "loss2.npy"
    loss3_file_path = output_dir + "/" + "loss3.npy"
    loss4_file_path = output_dir + "/" + "loss4.npy"

    if os.path.isfile(loss0_file_path) and os.path.isfile(loss1_file_path) and os.path.isfile(loss2_file_path) and os.path.isfile(loss3_file_path) and os.path.isfile(loss4_file_path):
        #loads the existing files
        loss0_file = np.load(loss0_file_path)
        loss1_file = np.load(loss1_file_path)
        loss2_file = np.load(loss2_file_path)
        loss3_file = np.load(loss3_file_path)
        loss4_file = np.load(loss4_file_path)
        #appends the new data to the array
        loss0_file = np.append(loss0_file,loss0)
        loss1_file = np.append(loss1_file,loss1)
        loss2_file = np.append(loss2_file,loss2)
        loss3_file = np.append(loss3_file,loss3)
        loss4_file = np.append(loss4_file,loss4)
        #saves the arrays
        np.save(loss0_file_path,loss0_file)
        np.save(loss1_file_path,loss1_file)
        np.save(loss2_file_path,loss2_file)
        np.save(loss3_file_path,loss3_file)
        np.save(loss4_file_path,loss4_file)
        print("Added data to numpy file")
    #if the file does not exist we create it
    else:
        loss0_file = []
        loss1_file = []
        loss2_file = []
        loss3_file = []
        loss4_file = []
        #adds the data to the lists
        loss0_file = np.append(loss0_file,loss0)
        loss0_file = np.append(loss0_file,loss0)
        loss0_file = np.append(loss0_file,loss0)
        loss0_file = np.append(loss0_file,loss0)
        loss0_file = np.append(loss0_file,loss0)
        #converts these lists to numpy arrays
        loss0_file = np.asarray(loss0_file)
        loss1_file = np.asarray(loss1_file)
        loss2_file = np.asarray(loss2_file)
        loss3_file = np.asarray(loss3_file)
        loss4_file = np.asarray(loss4_file)
        #saves the arrays
        np.save(loss0_file_path,loss0_file)
        np.save(loss1_file_path,loss1_file)
        np.save(loss2_file_path,loss2_file)
        np.save(loss3_file_path,loss3_file)
        np.save(loss4_file_path,loss4_file)





