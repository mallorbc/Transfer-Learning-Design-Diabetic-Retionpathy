import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout,Input,concatenate
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import time



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
    model.compile(optimizer=Adam(lr=0.00001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    #model.summary()

    return model

def inception_v3_multiple_inputs(image_width,image_height):
    input_shape = (image_width,image_height,3)
    image_input1 = Input(shape = input_shape)
    image_input2 = Input(shape = input_shape)

    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    input1 = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_tensor = image_input1, pooling='avg')
    input2 = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_tensor = image_input2, pooling='avg')
    # input1 = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(image_width, image_height, 3))
    # input2 = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(image_width, image_height, 3))
    input1.trainable = True
    input2.trainable = True
    
    #change the second branches layer names
    for layer in input2.layers:
        layer._name = layer.name + str("_2")
        
    layer1_1 = input1.output
    layer2_1 = input2.output
    # layer1_1 = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(layer1_1)
    # layer2_1 = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(layer2_1)


    flat1 = layers.Flatten()(layer1_1)
    flat2 = layers.Flatten()(layer2_1)

    concat_layer = concatenate([flat1,flat2])

    # output = layers.Flatten()(concat_layer)
    output = layers.Dense(1024,activation='relu')(concat_layer)
    output = Dropout(0.5)(output)
    output = layers.Dense(512,activation='relu')(output)
    output = Dropout(0.5)(output)
    # output = layers.Dense(256,activation='relu')(output)
    # output = Dropout(0.5)(output)
    # output = layers.Dense(128,activation='relu')(output)
    # output = Dropout(0.5)(output)
    # output = layers.Dense(64,activation='relu')(output)
    # output = Dropout(0.5)(output)
    output = layers.Dense(5,activation='softmax')(output)
    final_model = models.Model(inputs=[image_input1, image_input2], outputs=output)
    final_model.summary()
    final_model.compile(optimizer=Adam(lr=0.000001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    time.sleep(2)
    #plot_model(final_model, to_file='model_plot2.png')
    # plot_model(final_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # quit()
    return final_model