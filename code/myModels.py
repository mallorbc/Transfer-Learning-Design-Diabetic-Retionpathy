import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout
import os
# from tensorflow.keras.applications import inception_v3

def create_CNN(new_image_width,new_image_height):
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

    return model


def load_model(path_to_model):
    print("Loading Model...")
    model_to_load = models.load_model(path_to_model)
    return model_to_load


def save_model(model_to_save,run_dir):
    current_dir = run_dir
    output_dir = current_dir + "/checkpoints"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    number_of_checkpoints = os.listdir(output_dir)
    number_of_checkpoints = len(number_of_checkpoints)
    new_checkpoint_number = number_of_checkpoints + 1
    model_name = "checkpoint" + str(new_checkpoint_number)
    model_name = output_dir + "/" + model_name
    model_to_save.save(model_name)
    print("Saved Checkpoint!")

def transfer_learning_model_inception_v3(new_image_width, new_image_height,is_trainable):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    #sets the inception_v3 model to not update its weights
    base_model.trainable = is_trainable
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    #base_model.summary()
    #adds the two layers for transfer learning
    model = models.Sequential([base_model,global_average_layer])
    #adds dense and dropout layers for final output
    model.add(layers.Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    #model.summary()

    return model

