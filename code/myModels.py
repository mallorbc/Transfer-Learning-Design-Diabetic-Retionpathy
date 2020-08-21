import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout,Input,concatenate
import os
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model
import numpy as np

import time

from preprocessData import *
from utils import *

from efficientnet.tfkeras import EfficientNetB6 as Net

import tensorflow_addons as tfa
import gc

import random
import matplotlib.pyplot as plt



def create_DenseNet(image_width,image_height,learning_rate=0.0001):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    # if is_trainable is None:
    #     is_trainable = True
    base_model = tf.keras.applications.densenet.DenseNet121(weights="imagenet",include_top=False,input_shape=(image_width, image_height, 3))
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = True
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model.output)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(5, activation='softmax')(model)
    final_model = models.Model(inputs=[base_model.input], outputs=output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    final_model.summary()
    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model


def create_CNN(image_width,image_height,learning_rate=0.000025):
    input_shape = (image_width,image_height,3)
    image_input = Input(shape = input_shape)
    output = simple_layer(32,3,image_input)
    output = simple_layer(64,3,output)
    output = simple_layer(128,3,output)
    output = simple_layer(256,3,output)
    output = simple_layer(512,3,output)
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    output = layers.Dense(1024)(output)
    output = layers.PReLU()(output)
    output = Dropout(0.5)(output)
    output = layers.Dense(512)(output)
    output = layers.PReLU()(output)
    output = Dropout(0.5)(output)
    output = layers.Dense(256)(output)
    output = layers.PReLU()(output)
    output = Dropout(0.5)(output)
    output = layers.Dense(5,activation='softmax')(output)


    final_model = models.Model(inputs=[image_input], outputs=output)
    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    final_model.summary()
    print("Learning rate is: " + str(learning_rate))


    time.sleep(2)
    #plot_model(final_model, to_file='model.png')
    return final_model



def load_model(path_to_model,model_number=None,width=None,height=None,frozen=None):
    print("Loading Model...")
    if model_number is None:
        model_to_load = models.load_model(path_to_model)
        model_to_load.summary()
        print("Loaded entire model")
    elif model_number == 1:
        model_to_load = create_CNN(width,height)
        model_to_load.load_weights(path_to_model)
        print("Loaded model weights")

    elif model_number == 2:
        model_to_load = transfer_learning_model_inception_v3_functional(width,height,frozen)
        model_to_load.load_weights(path_to_model)
        print("Loaded model weights")

    elif model_number == 3:
        model_to_load = inception_v3_multiple_inputs(width,height)
        model_to_load.load_weights(path_to_model)
        print("Loaded model weights")
    elif model_number == 4:
        model_to_load = transfer_learning_model_inception_v3(width,height,frozen)
        model_to_load.load_weights(path_to_model)
        print("Loaded model weights")
    elif model_number == 5:
        model_to_load = efficientnet(width, height)
        model_to_load.load_weights(path_to_model)
        print("Loaded model weights")
    elif model_number == 7:
        temp_binary_model = inception_v3_functional_binary_no_act(width,height)
        temp_non_zero_model = inception_v3_non_zero_no_act(width,height)
        model_to_load = dual_inception_v3_ensemble(width,height,temp_binary_model,temp_non_zero_model)
        model_to_load.load_weights(path_to_model)
        print("Loaded model weights")
    elif model_number == 8:
        model_to_load = inception_v3_non_zero(width, height)
        model_to_load.load_weights(path_to_model)


    #supporting model for model 7
    elif model_number == 77:
        model_to_load = inception_v3_functional_binary_no_act(width, height)
        model_to_load.load_weights(path_to_model)
    elif model_number == 78:
        model_to_load = inception_v3_non_zero_no_act(width,height)
        model_to_load.load_weights(path_to_model)
    return model_to_load


def save_model(model_to_save,run_dir,whole_model=None):
    gc.collect()
    current_dir = run_dir
    output_dir = current_dir + "/checkpoints"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    number_of_checkpoints = list(os.walk(output_dir))
    number_of_checkpoints = len(number_of_checkpoints[0][1])
    new_checkpoint_number = number_of_checkpoints + 1
    model_name = "checkpoint" + str(new_checkpoint_number)
    model_name = output_dir + "/" + model_name
    os.makedirs(model_name)
    if whole_model is None:
        model_name = model_name + "/savepoint.h5"
        # tf.keras.models.save_model(model_to_save,model_name)
        # tf.keras.backend.clear_session()
        # model_to_save.save_model(model_name)
        model_to_save.save_weights(model_name)
        print("Saved Model Weights!")
    else:
        model_name = model_name + "/whole_model"
        tf.keras.models.save_model(model_to_save,model_name)
        print("Saved Entire Model!")

    print("Saved Checkpoint!")


def inception_v3_multiple_inputs(image_width,image_height,learning_rate=0.00001):
    input_shape = (image_width,image_height,3)
    image_input1 = Input(shape = input_shape)
    image_input2 = Input(shape = input_shape)

    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    input1 = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(image_width, image_height, 3))
    input2 = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(image_width, image_height, 3))
    input1.trainable = True
    input2.trainable = True
    
    #change the second branches layer names
    for layer in input2.layers:
        layer._name = layer.name + str("_2")
        
    layer1_1 = input1.output
    layer2_1 = input2.output
    layer1_1 = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(layer1_1)
    layer2_1 = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool2')(layer2_1)

    concat_layer = concatenate([layer1_1,layer2_1],axis=-1)

    output = layers.Dense(1024)(concat_layer)
    output = layers.PReLU()(output)
    output = Dropout(0.5)(output)
    output = layers.Dense(512)(output)
    output = layers.PReLU()(output)
    output = Dropout(0.5)(output)
    output = layers.Dense(256)(output)
    output = layers.PReLU()(output)
    output = Dropout(0.5)(output)



    output = layers.Dense(5,activation='softmax')(output)
    final_model = models.Model(inputs=[input1.input, input2.input], outputs=output)
    # final_model.summary()
    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    # final_model.compile(optimizer=Adam(lr=0.000001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    #plot_model(final_model, to_file='model_plot2.png')
    # plot_model(final_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    final_model.summary()
    print("The learning rate is " + str(learning_rate))

    return final_model

def efficientnet(new_image_width, new_image_height,learning_rate=0.0001):
    base_model = Net(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))
    base_model.trainable = True
    # base_model.trainable = is_trainable
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model.output)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(5, activation='softmax')(model)
    final_model = models.Model(inputs=[base_model.input], outputs=output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    final_model.summary()
    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model


def inception_v3_resnet(image_width,image_height):
    input_shape = (image_width,image_height,3)
    image_input1 = Input(shape = input_shape)
    image_input2 = Input(shape = input_shape)

    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    input1 = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_tensor = image_input1, pooling='avg')
    input2 = tf.keras.applications.ResNet50(weights="imagenet",include_top=False,input_tensor = image_input2, pooling='avg')
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
    return final_model


def get_model_predictions_one_input(loaded_model,images):
    print(len(images))
    total_predictions = []
    total_predictions = np.asarray(total_predictions)

    test_size = 400
    number_of_large_groups = math.floor(len(images)/test_size)

    counter = 1
    for i in range(number_of_large_groups):
        image_batch = images[((counter - 1)*test_size):((counter*test_size))]
        image_batch = normalize_images(image_batch)
        # print(image_batch)
        # quit()
        image_batch = np.asarray(image_batch)
        print(str(counter*test_size) + " images tested")
        # total_predictions = np.append(total_predictions,loaded_model.predict_classes(image_batch))
        total_predictions = np.append(total_predictions,custom_predict_class(loaded_model,image_batch))
        print("total predictions:",np.shape(total_predictions))
        if (counter*test_size)>=12500:
            return total_predictions

        # print("total labels: ",np.shape(total_labels))
        counter = counter + 1
        gc.collect()

    #makes the rest of the predictions
    image_batch = images[(test_size*number_of_large_groups):(len(images))]
    image_batch = normalize_images(image_batch)
    image_batch = np.asarray(image_batch)

    total_predictions = np.append(total_predictions,custom_predict_class(loaded_model,image_batch))

    return total_predictions

def get_model_predictions_two_inputs(loaded_model,image_one,image_two):
    total_predictions = []
    total_predictions = np.asarray(total_predictions)

    test_size = 50
    number_of_large_groups = math.floor(len(image_one)/test_size)

    counter = 1
    for i in range(number_of_large_groups):
        image_batch_one = image_one[((counter - 1)*test_size):((counter*test_size))]
        image_batch_one = normalize_images(image_batch_one)
        image_batch_one = np.asarray(image_batch_one)

        image_batch_two = image_two[((counter - 1)*test_size):((counter*test_size))]
        image_batch_two = normalize_images(image_batch_two)
        image_batch_two = np.asarray(image_batch_two)
        print(str(counter*test_size) + " images tested")
        total_predictions = np.append(total_predictions,loaded_model.predict_classes([image_batch_one,image_batch_two]))
        print("total predictions:",np.shape(total_predictions))

        # print("total labels: ",np.shape(total_labels))
        counter = counter + 1
    
    #makes the rest of the predictions
    image_batch_one = image_one[(test_size*number_of_large_groups):(len(image_one))]
    image_batch_one = normalize_images(image_batch_one)
    image_batch_one = np.asarray(image_batch_one)

    image_batch_two = image_one[(test_size*number_of_large_groups):(len(image_one))]
    image_batch_two = normalize_images(image_batch_two)
    image_batch_two = np.asarray(image_batch_two)
    total_predictions = np.append(total_predictions,loaded_model.predict_classes([image_batch_one,image_batch_two]))

    return total_predictions



def simple_layer(filters,number_of_layers,input_layer = None,prelu=None):
    if prelu is None:
        output = layers.Conv2D(filters, (3, 3),strides=1,activation="relu",padding="same")(input_layer)        
        if number_of_layers>=2:
            for i in range(number_of_layers-2):
                output = layers.Conv2D(filters, (3, 3),strides=1,activation="relu",padding="same")(output)
            output = layers.Conv2D(filters, (3, 3),strides=2,activation="relu",padding="same")(output)
    else:
        output = layers.Conv2D(filters, (3, 3),strides=1,padding="same")(input_layer)
        output = layers.PReLU()(output)
        if number_of_layers>=2:
            for i in range(number_of_layers-2):
                output = layers.Conv2D(filters, (3, 3),strides=1,padding="same")(output)
                output = layers.PReLU()(output)
            output = layers.Conv2D(filters, (3, 3),strides=2,padding="same")(output)
            output = layers.PReLU()(output)



    return output

def custom_predict_class(model,image_to_test):
    return_values = []
    count = 0
    for image in image_to_test:
        # image = cv2.imread(image)
        # image = np.asarray(image)
        # image.reshape(512,512,3)
        image = np.expand_dims(image,axis=0)
        # image = image.astype("float32")
        prediction_array = model.predict(image)
        prediction_array = np.squeeze(prediction_array,axis=0)
        prediction_class = np.where(prediction_array == np.amax(prediction_array))
        # prediction_class = int(prediction_class)
        prediction_class = np.asarray(prediction_class)
        prediction_class = np.squeeze(prediction_class,axis=0)
        print(prediction_class)
        return_values.append(prediction_class)
    return return_values


def transfer_learning_model_inception_v3_functional(new_image_width, new_image_height,is_trainable=True,learning_rate=0.0001):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    if is_trainable is None:
        is_trainable = True
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = is_trainable
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model.output)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(5, activation='softmax')(model)
    final_model = models.Model(inputs=[base_model.input], outputs=output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    final_model.summary()
    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model

def inception_v3_non_zero(new_image_width, new_image_height,is_trainable=True,learning_rate=0.00001):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    if is_trainable is None:
        is_trainable = True
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = is_trainable
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model.output)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(4, activation='softmax')(model)
    final_model = models.Model(inputs=[base_model.input], outputs=output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    final_model.summary()
    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model

def inception_v3_non_zero_no_act(new_image_width, new_image_height,is_trainable=True,learning_rate=0.00001):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    if is_trainable is None:
        is_trainable = True
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = is_trainable
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model.output)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(4)(model)
    final_model = models.Model(inputs=[base_model.input], outputs=output)


    return final_model


def inception_v3_functional_binary(new_image_width, new_image_height,is_trainable=True,learning_rate=0.00001):
    if is_trainable is None:
        is_trainable = True
    input_shape = (new_image_width,new_image_height,3)
    image_input = Input(shape = input_shape)
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = is_trainable
    base_model = base_model(image_input)
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(1,activation="sigmoid")(model)
    final_model = models.Model(inputs=[image_input], outputs=output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    final_model.summary()
    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model


def inception_v3_functional_binary_no_act(new_image_width, new_image_height,is_trainable=True,learning_rate=0.0001):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    if is_trainable is None:
        is_trainable = True
    input_shape = (new_image_width,new_image_height,3)
    image_input = Input(shape = input_shape)
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = True
    base_model = base_model(image_input)
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(1)(model)
    final_model = models.Model(inputs=[image_input], outputs=output)

    return final_model

def dual_inception_v3_functional_binary_based(new_image_width, new_image_height,binary_model,is_trainable=True,learning_rate=0.00005):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    input_shape = (new_image_width,new_image_height,3)
    image_input = Input(shape = input_shape)
    if is_trainable is None:
        is_trainable = True
    # base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))

    for layer in base_model.layers:
        layer._name = layer.name + str("_2")
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = is_trainable
    binary_model.trainable = True
    base_model = base_model(image_input)
    binary_model = binary_model(image_input)

    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(4)(model)

    output = concatenate([binary_model,output])
    output = layers.PReLU()(output)
    output = layers.Dense(5,activation="softmax")(output)

    # output = layers.Activation("softmax")(output)

    # output = layers.Dense(1, activation='sigmoid')(model)
    final_model = models.Model(inputs=[image_input], outputs=output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    final_model.summary()

    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model

def dual_inception_v3_functional_binary_based2(new_image_width, new_image_height,binary_model,is_trainable=True,learning_rate=0.00005):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    input_shape = (new_image_width,new_image_height,3)
    image_input = Input(shape = input_shape)
    if is_trainable is None:
        is_trainable = True
    # base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))

    for layer in base_model.layers:
        layer._name = layer.name + str("_2")
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = is_trainable
    binary_model.trainable = True
    base_model = base_model(image_input)
    binary_model = binary_model(image_input)

    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = global_average_layer(base_model)
    model = layers.Dense(1024)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model =  layers.Dense(512)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    model = layers.Dense(256)(model)
    model = layers.PReLU()(model)
    model = Dropout(0.5)(model)
    output = layers.Dense(4)(model)

    output1 = concatenate([binary_model,output])
    output1 = layers.Dense(4)(output1)
    output1 = layers.PReLU()(output1)
    # output1 = Dropout(0.5)(output1)

    output = concatenate([binary_model,output1])

    # output = layers.PReLU()(output)
    output = layers.Dense(5,activation="softmax")(output)

    # output = layers.Activation("softmax")(output)

    # output = layers.Dense(1, activation='sigmoid')(model)
    final_model = models.Model(inputs=[image_input], outputs=output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    final_model.summary()

    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model

def dual_inception_v3_ensemble(new_image_width, new_image_height,binary_model,non_zero_model,is_trainable=True,learning_rate=0.0001):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    input_shape = (new_image_width,new_image_height,3)
    image_input = Input(shape = input_shape)
    #sets the inception_v3 model to not update its weights
    non_zero_model.trainable = False
    binary_model.trainable = False
    # non_zero_model.trainable = True
    # binary_model.trainable = True
    non_zero_model = non_zero_model(image_input)
    binary_model = binary_model(image_input)

    output1 = concatenate([binary_model,non_zero_model])
    final_output = layers.PReLU()(output1)
    # final_output = layers.Dropout(0.5)(final_output)
    final_output = layers.Dense(5)(final_output)
    final_output = layers.PReLU()(final_output)
    # final_output = layers.Dropout(0.5)(final_output)
    final_output = layers.Dense(5)(final_output)
    final_output = layers.PReLU()(final_output)
    # final_output = layers.Dropout(0.5)(final_output)
    final_output = layers.Dense(5,activation="softmax")(final_output)

    final_model = models.Model(inputs=[image_input], outputs=final_output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    final_model.summary()

    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model

def custom_predict_class(model,image_to_test):
    return_values = []
    for image in image_to_test:
        image = np.expand_dims(image,axis=0)
        prediction_array = model.predict(image)
        prediction_array = np.squeeze(prediction_array,axis=0)
        prediction_class = np.where(prediction_array == np.amax(prediction_array))
        # prediction_class = int(prediction_class)
        prediction_class = np.asarray(prediction_class)
        prediction_class = np.squeeze(prediction_class,axis=0)
        return_values.append(prediction_class)
    return return_values

def get_loss_acc_of_each_class(model,images,width,height,test_size,health_dict):
    images = get_all_images_of_one_class(0,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,0)
    print("Testing all images in class 0...")
    loss_0,acc_0 = evaluate_all_images(model,images,labels,test_size)


    images = get_all_images_of_one_class(1,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,1)
    print("Testing all images in class 1...")
    loss_1,acc_1 = evaluate_all_images(model,images,labels,test_size)


    images = get_all_images_of_one_class(2,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,2)
    print("Testing all images in class 2...")
    loss_2,acc_2 = evaluate_all_images(model,images,labels,test_size)


    images = get_all_images_of_one_class(3,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,3)
    print("Testing all images in class 3...")
    loss_3,acc_3 = evaluate_all_images(model,images,labels,test_size)

    images = get_all_images_of_one_class(4,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,4)
    print("Testing all images in class 4...")
    loss_4,acc_4 = evaluate_all_images(model,images,labels,test_size)

    return loss_0,loss_1,loss_2,loss_3,loss_4,acc_0,acc_1,acc_2,acc_3,acc_4


def get_loss_acc_of_each_class_binary(model,images,width,height,test_size,health_dict):
    images = get_num_images_of_one_class(0,images,health_dict,test_size)
    image_batch = normalize_images(images)
    np_image_batch = np.asarray(image_batch)
    np_image_batch.reshape(len(image_batch),width,height,3)
    labels = np.ones(len(images))
    labels = np.multiply(labels,0)
    metrics = model.evaluate(np_image_batch,labels,verbose=0)
    loss_0 = metrics[0]
    acc_0 = metrics[-1]

    images = get_num_images_of_one_class(1,images,health_dict,test_size)
    image_batch = normalize_images(images)
    np_image_batch = np.asarray(image_batch)
    np_image_batch.reshape(len(image_batch),width,height,3)
    labels = np.ones(len(images))
    labels = np.multiply(labels,1)
    metrics = model.evaluate(np_image_batch,labels,verbose=0)
    loss_1 = metrics[0]
    acc_1 = metrics[-1]


    return loss_0,loss_1,acc_0,acc_1

def get_loss_acc_of_each_class_binary2(model,images,width,height,test_size,health_dict):
    images = get_all_images_of_one_class(0,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,0)
    print("Testing all images in class 0...")
    loss_0,acc_0 = evaluate_all_images(model,images,labels,test_size)


    images = get_all_images_of_one_class(1,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,1)
    print("Testing all images in class 1...")
    loss_1,acc_1 = evaluate_all_images(model,images,labels,test_size)


    return loss_0,loss_1,acc_0,acc_1

def get_loss_acc_of_each_class_non_zero(model,images,width,height,test_size,health_dict):
    images = get_all_images_of_one_class(0,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,0)
    print("Testing all images in class 0...")
    loss_0,acc_0 = evaluate_all_images(model,images,labels,test_size)


    images = get_all_images_of_one_class(1,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,1)
    print("Testing all images in class 1...")
    loss_1,acc_1 = evaluate_all_images(model,images,labels,test_size)


    images = get_all_images_of_one_class(2,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,2)
    print("Testing all images in class 2...")
    loss_2,acc_2 = evaluate_all_images(model,images,labels,test_size)


    images = get_all_images_of_one_class(3,health_dict)
    labels = np.ones(len(images))
    labels = np.multiply(labels,3)
    print("Testing all images in class 3...")
    loss_3,acc_3 = evaluate_all_images(model,images,labels,test_size)


    return loss_0,loss_1,loss_2,loss_3,acc_0,acc_1,acc_2,acc_3

def adjust_class_weights(loss0,loss1,loss2,loss3,loss4):
    classes = [0,1,2,3,4]
    losses = [] 
    losses.append(loss0)
    losses.append(loss1)
    losses.append(loss2)
    losses.append(loss3)
    losses.append(loss4)
    losses = np.asarray(losses)
    losses = np.multiply(losses,5)
    # zipped_list = sorted(zip(classes,losses),key=lambda x:x[1])
    zipped_list = zip(losses,classes)
    # print(zipped_list)
    zipped_list = sorted(zipped_list)
    # print(zipped_list)
    # print(zipped_list[-1][0])
    # quit()
    # print(zipped_list[-1][-1])
    highest_loss = zipped_list[-1][0]
    # print(zipped_list[-1])
    # print(zipped_list)
    # losses = np.asarray(losses)
    # print("Highest loss was " + str(highest_loss) + " for class " + str(zipped_list[-1][-1]))
    losses = np.divide(losses,highest_loss)
    # print(losses)
    # quit()
    class_weights_list = losses
    class_weights = {0: class_weights_list[0], 1: class_weights_list[1], 2: class_weights_list[2], 3: class_weights_list[3], 4: class_weights_list[4]}
    print("New class weights: ",class_weights)
    return class_weights

def adjust_base_weights(loss0,loss1,loss2,loss3,loss4,base0,base1,base2,base3,base4,run_dir):
    classes = [0,1,2,3,4]
    losses = [] 
    adjusted_base_weights = []
    #adds the losses to a np array to allow easy math
    losses.append(loss0)
    losses.append(loss1)
    losses.append(loss2)
    losses.append(loss3)
    losses.append(loss4)
    losses = np.asarray(losses)
    losses = np.multiply(losses,5)
    zipped_list = zip(losses,classes)
    zipped_list = sorted(zipped_list)
    highest_loss = zipped_list[-1][0]
    lowest_loss = zipped_list[0][0]
    losses = np.divide(losses,highest_loss)
    print(losses)
    adjusted_base_weights.append(losses[0]*base0)
    adjusted_base_weights.append(losses[1]*base1)
    adjusted_base_weights.append(losses[2]*base2)
    adjusted_base_weights.append(losses[3]*base3)
    adjusted_base_weights.append(losses[4]*base4)

    class_weights_list = adjusted_base_weights
    class_weights = {0: class_weights_list[0], 1: class_weights_list[1], 2: class_weights_list[2], 3: class_weights_list[3], 4: class_weights_list[4]}
    print("New class weights: ",class_weights)
    return class_weights

def adjust_base_weights_binary(loss0,loss1,base0,base1,run_dir):
    classes = [0,1]
    losses = [] 
    adjusted_base_weights = []
    #adds the losses to a np array to allow easy math
    losses.append(loss0)
    losses.append(loss1)
    losses = np.asarray(losses)
    losses = np.multiply(losses,2)
    zipped_list = zip(losses,classes)
    zipped_list = sorted(zipped_list)
    highest_loss = zipped_list[-1][0]
    lowest_loss = zipped_list[0][0]
    losses = np.divide(losses,highest_loss)
    print("loss ratios: ",losses)
    adjusted_base_weights.append(base0 + losses[0]*base0)
    adjusted_base_weights.append(base1 + losses[1]*base1)


    class_weights_list = adjusted_base_weights
    class_weights = {0: class_weights_list[0], 1: class_weights_list[1]}
    print("New class weights: ",class_weights)
    return class_weights

def adjust_base_weights_test(loss0,base0,loss1,base1,run_dir,loss2=None,base2=None,loss3=None,base3=None,loss4=None,base4=None):
    classes = [0,1]
    losses = [] 
    adjusted_base_weights = []
    #adds the losses to a np array to allow easy math
    losses.append(loss0)
    losses.append(loss1)
    if loss2 is not None:
        losses.append(loss2)
    if loss3 is not None:
        losses.append(loss3)
    if loss4 is not None:
        losses.append(loss4)
    losses = np.asarray(losses)
    # losses = np.multiply(losses,2)
    zipped_list = zip(losses,classes)
    zipped_list = sorted(zipped_list)
    highest_loss = zipped_list[-1][0]
    # lowest_loss = zipped_list[0][0]
    print(losses)
    losses = np.divide(losses,highest_loss)
    print("loss ratios: ",losses)
    if loss2 is None:
        adjusted_base_weights.append(base0 + losses[0]*base0)
        adjusted_base_weights.append(base1 + losses[1]*base1)
        class_weights_list = adjusted_base_weights
        class_weights = {0: class_weights_list[0], 1: class_weights_list[1]}
    elif loss4 is None:
        adjusted_base_weights.append(base0 + (3*losses[0]*base0))
        adjusted_base_weights.append(base1 + (3*losses[1]*base1))
        adjusted_base_weights.append(base2 + (3*losses[2]*base2))
        adjusted_base_weights.append(base3 + (3*losses[3]*base3))
        class_weights_list = adjusted_base_weights
        class_weights = {0: class_weights_list[0], 1: class_weights_list[1], 2: class_weights_list[2], 3: class_weights_list[3]}
    else:
        adjusted_base_weights.append(base0 + (4*losses[0]*base0))
        adjusted_base_weights.append(base1 + (4*losses[1]*base1))
        adjusted_base_weights.append(base2 + (4*losses[2]*base2))
        adjusted_base_weights.append(base3 + (4*losses[3]*base3))
        adjusted_base_weights.append(base4 + (4*losses[4]*base4))
        class_weights_list = adjusted_base_weights
        class_weights = {0: class_weights_list[0], 1: class_weights_list[1], 2: class_weights_list[2], 3: class_weights_list[3], 4: class_weights_list[4]}
    

    print("New class weights: ",class_weights)
    # print(highest_loss)
    # quit()
    return class_weights

def adjust_base_weights2(loss0,loss1,loss2,loss3,loss4,base0,base1,base2,base3,base4,run_dir):
    # add_class_loss_data(loss0,loss1,loss2,loss3,loss4,run_dir)
    classes = [0,1,2,3,4]
    losses = [] 
    adjusted_base_weights = []
    #adds the losses to a np array to allow easy math
    losses.append(loss0)
    losses.append(loss1)
    losses.append(loss2)
    losses.append(loss3)
    losses.append(loss4)
    losses = np.asarray(losses)
    losses = np.multiply(losses,5)
    zipped_list = zip(losses,classes)
    zipped_list = sorted(zipped_list)
    highest_loss = zipped_list[-1][0]
    lowest_loss = zipped_list[0][0]
    # losses = np.subtract(losses,lowest_loss)
    losses = np.divide(losses,highest_loss)
    print(losses)

    adjusted_base_weights.append(base0 + (4*losses[0]*base0))
    adjusted_base_weights.append(base1 + (4*losses[1]*base1))
    adjusted_base_weights.append(base2 + (4*losses[2]*base2))
    adjusted_base_weights.append(base3 + (4*losses[3]*base3))
    adjusted_base_weights.append(base4 + (4*losses[4]*base4))
    class_weights_list = adjusted_base_weights
    class_weights = {0: class_weights_list[0], 1: class_weights_list[1], 2: class_weights_list[2], 3: class_weights_list[3], 4: class_weights_list[4]}
    print("New class weights: ",class_weights)
    return class_weights

def evaluate_all_images(model,images,labels,test_size):
    # print("Testing all images")
    total_loss = []
    total_acc = []
    total_loss = np.asarray(total_loss)
    total_acc = np.asarray(total_acc)
    number_of_large_groups = math.floor(len(images)/test_size)
    if number_of_large_groups > 0:
        counter = 1
        images_tested = len(images)
        #creates the list of losses and accuracy
        for i in range(number_of_large_groups):
            image_batch = images[((counter - 1)*test_size):((counter*test_size))]
            image_batch = normalize_images(image_batch)
            label_batch = labels[((counter - 1)*test_size):((counter*test_size))]
            image_batch = np.asarray(image_batch)
            label_batch = np.asarray(label_batch)
            metrics = model.evaluate(image_batch,label_batch,verbose=0)
            total_loss = np.append(total_loss,metrics[0])
            total_acc = np.append(total_acc,metrics[-1])
            counter = counter + 1
            memory_used = get_memory_usage_percentage()
            if memory_used>40:
                #trys to save some memory
                gc.collect()
            if memory_used>85:
                print("memory used exceeded 85 percent")
                print("tested "+str(len(total_loss)*test_size) + " images")
                images_tested = len(total_loss)*test_size
                break
        #finds the average loss and accuracy
        average_loss = np.mean(total_loss)
        average_acc = np.mean(total_acc)
        #tests the leftover images
        leftover_images = images[(test_size*number_of_large_groups):(len(images))]
        leftover_labels = labels[(test_size*number_of_large_groups):(len(images))]
        #if there are leftover images, test them
        if len(leftover_images) > 0:
            image_batch = normalize_images(leftover_images)
            image_batch = np.asarray(image_batch)
            label_batch = np.asarray(leftover_labels)
            final_metrics = model.evaluate(image_batch,label_batch,verbose=0)
            final_metrics_loss = final_metrics[0]
            final_metrics_acc = final_metrics[-1]
            #averages the loss proportional to size
            percent_leftover = len(leftover_images)/images_tested
            percent_large_groups = 1 -percent_leftover
            all_loss = (average_loss * percent_large_groups) + (final_metrics_loss * percent_leftover)
            all_acc = (average_acc * percent_large_groups) + (final_metrics_acc * percent_leftover)
        else:
            all_loss = average_loss
            all_acc = average_acc
        #clears memory
        gc.collect()
    else:
        image_batch = normalize_images(images)
        image_batch = np.asarray(image_batch)
        label_batch = np.asarray(labels)
        final_metrics = model.evaluate(image_batch,label_batch,verbose=0)
        all_loss = final_metrics[0]
        all_acc = final_metrics[-1]
    return all_loss, all_acc

def evaluate_all_images_two_inputs(model,images,images2,labels,test_size):
    # print("Testing all images")
    total_loss = []
    total_acc = []
    total_loss = np.asarray(total_loss)
    total_acc = np.asarray(total_acc)
    number_of_large_groups = math.floor(len(images)/test_size)
    if number_of_large_groups > 0:
        counter = 1
        images_tested = len(images)
        #creates the list of losses and accuracy
        for i in range(number_of_large_groups):
            image_batch = images[((counter - 1)*test_size):((counter*test_size))]
            image_batch2 = images2[((counter - 1)*test_size):((counter*test_size))]
            image_batch = normalize_images(image_batch)
            image_batch2 = normalize_images(image_batch2)

            label_batch = labels[((counter - 1)*test_size):((counter*test_size))]
            image_batch = np.asarray(image_batch)
            image_batch2 = np.asarray(image_batch2)
            label_batch = np.asarray(label_batch)
            metrics = model.evaluate([image_batch,image_batch2], label_batch,verbose=0)
            total_loss = np.append(total_loss,metrics[0])
            total_acc = np.append(total_acc,metrics[-1])
            counter = counter + 1
            memory_used = get_memory_usage_percentage()
            if memory_used>40:
                #trys to save some memory
                gc.collect()
            if memory_used>85:
                print("memory used exceeded 85 percent")
                print("tested "+str(len(total_loss)*test_size) + " images")
                images_tested = len(total_loss)*test_size
                break
        #finds the average loss and accuracy
        average_loss = np.mean(total_loss)
        average_acc = np.mean(total_acc)
        #tests the leftover images
        leftover_images = images[(test_size*number_of_large_groups):(len(images))]
        leftover_images2 = images2[(test_size*number_of_large_groups):(len(images2))]

        leftover_labels = labels[(test_size*number_of_large_groups):(len(images))]
        #if there are leftover images, test them
        if len(leftover_images) > 0:
            image_batch = normalize_images(leftover_images)
            image_batch2 =normalize_images(leftover_images2)
            image_batch = np.asarray(image_batch)
            image_batch2 = np.asarray(image_batch2)
            label_batch = np.asarray(leftover_labels)
            final_metrics = model.evaluate([image_batch,image_batch2], label_batch,verbose=0)
            final_metrics_loss = final_metrics[0]
            final_metrics_acc = final_metrics[-1]
            #averages the loss proportional to size
            percent_leftover = len(leftover_images)/images_tested
            percent_large_groups = 1 -percent_leftover
            all_loss = (average_loss * percent_large_groups) + (final_metrics_loss * percent_leftover)
            all_acc = (average_acc * percent_large_groups) + (final_metrics_acc * percent_leftover)
        else:
            all_loss = average_loss
            all_acc = average_acc
        #clears memory
        gc.collect()
    else:
        image_batch = normalize_images(images)
        image_batch2 = normalize_images(images2)
        image_batch = np.asarray(image_batch)
        image_batch2 = np.asarray(image_batch2)
        label_batch = np.asarray(labels)
        final_metrics = model.evaluate([image_batch,image_batch2], label_batch,verbose=0)
        all_loss = final_metrics[0]
        all_acc = final_metrics[-1]
    return all_loss, all_acc

def evaluate_all_images_each_class(model,health_dict,test_size):
    #initializes variables
    num_two = 0
    num_three = 0
    num_four = 0
    loss2_partial = 0
    loss3_partial = 0
    loss4_partial = 0
    acc2_partial = 0
    acc3_partial = 0
    acc4_partial = 0
    total_loss,total_acc,loss0,acc0,loss1,acc1,loss2,acc2,loss3,acc3,loss4,acc4 = 0,0,0,0,0,0,0,0,0,0,0,0




    cat_num = len(health_dict)
    zero_images = health_dict[0]
    one_images = health_dict[1]
    two_images = health_dict[2]
    three_images = health_dict[3]
    four_images = health_dict[4]

    if len(four_images)>0:
        cat_num = 5
    elif len(three_images)>0:
        cat_num = 4
    elif len(two_images)>0:
        cat_num = 3
    else:
        cat_num = 2

    zero_images = health_dict[0]
    num_zero = len(zero_images)
    zero_labels = np.ones(len(zero_images))
    zero_labels = np.multiply(zero_labels,0)
    print("Testing all images in class 0...")
    loss0,acc0 = evaluate_all_images(model,zero_images,zero_labels,test_size)

    one_images = health_dict[1]
    num_one = len(one_images)
    one_labels = np.ones(len(one_images))
    one_labels = np.multiply(one_labels,1)
    print("Testing all images in class 1...")
    loss1,acc1 = evaluate_all_images(model,one_images,one_labels,test_size)
    if cat_num >2:
        two_images = health_dict[2]
        num_two = len(two_images)
        two_labels = np.ones(len(two_images))
        two_labels = np.multiply(two_labels,2)
        print("Testing all images in class 2...")
        loss2,acc2 = evaluate_all_images(model,two_images,two_labels,test_size)

        three_images = health_dict[3]
        num_three = len(three_images)
        three_labels = np.ones(len(three_images))
        three_labels = np.multiply(three_labels,3)
        print("Testing all images in class 3...")
        loss3,acc3 = evaluate_all_images(model,three_images,three_labels,test_size)
    if cat_num == 5:
        four_images = health_dict[4]
        num_four = len(four_images)
        four_labels = np.ones(len(four_images))
        four_labels = np.multiply(four_labels,4)
        print("Testing all images in class 4...")
        loss4,acc4 = evaluate_all_images(model,four_images,four_labels,test_size)

    num_total = num_zero + num_one + num_two + num_three + num_four

    percent_zero = num_zero/num_total
    loss0_partial = loss0*percent_zero
    acc0_partial = acc0 * percent_zero

    percent_one = num_one/num_total
    loss1_partial = loss1*percent_one
    acc1_partial = acc1 * percent_one

    if cat_num>2:
        pecent_two = num_two/num_total
        loss2_partial = loss2 * pecent_two
        acc2_partial = acc2 * pecent_two
        
        percent_three = num_three/num_total
        loss3_partial = loss3 * percent_three
        acc3_partial = acc3 * percent_three
    if cat_num == 5:
        percent_four = num_four/num_total
        loss4_partial = loss4 * percent_four
        acc4_partial = acc4 * percent_four

    total_loss = loss0_partial + loss1_partial + loss2_partial + loss3_partial + loss4_partial
    total_acc = acc0_partial + acc1_partial + acc2_partial + acc3_partial + acc4_partial
    
    return total_loss,total_acc,loss0,acc0,loss1,acc1,loss2,acc2,loss3,acc3,loss4,acc4


def evaluate_all_images_each_class_two_inputs(model,health_dict,health_dict2,test_size):
    #initializes variables
    num_two = 0
    num_three = 0
    num_four = 0
    loss2_partial = 0
    loss3_partial = 0
    loss4_partial = 0
    acc2_partial = 0
    acc3_partial = 0
    acc4_partial = 0
    total_loss,total_acc,loss0,acc0,loss1,acc1,loss2,acc2,loss3,acc3,loss4,acc4 = 0,0,0,0,0,0,0,0,0,0,0,0




    cat_num = len(health_dict)
    zero_images = health_dict[0]
    one_images = health_dict[1]
    two_images = health_dict[2]
    three_images = health_dict[3]
    four_images = health_dict[4]

    if len(four_images)>0:
        cat_num = 5
    elif len(three_images)>0:
        cat_num = 4
    elif len(two_images)>0:
        cat_num = 3
    else:
        cat_num = 2

    zero_images = health_dict[0]
    zero_images2 = health_dict2[0]
    num_zero = len(zero_images)
    zero_labels = np.ones(len(zero_images))
    zero_labels = np.multiply(zero_labels,0)
    print("Testing all images in class 0...")
    loss0,acc0 = evaluate_all_images_two_inputs(model,zero_images,zero_images2,zero_labels,test_size)

    one_images = health_dict[1]
    one_images2 = health_dict2[1]
    num_one = len(one_images)
    one_labels = np.ones(len(one_images))
    one_labels = np.multiply(one_labels,1)
    print("Testing all images in class 1...")
    loss1,acc1 = evaluate_all_images_two_inputs(model,one_images,one_images2,one_labels,test_size)
    if cat_num >2:
        two_images = health_dict[2]
        two_images2 = health_dict2[2]
        num_two = len(two_images)
        two_labels = np.ones(len(two_images))
        two_labels = np.multiply(two_labels,2)
        print("Testing all images in class 2...")
        loss2,acc2 = evaluate_all_images_two_inputs(model,two_images,two_images2,two_labels,test_size)

        three_images = health_dict[3]
        three_images2 = health_dict2[3]
        num_three = len(three_images)
        three_labels = np.ones(len(three_images))
        three_labels = np.multiply(three_labels,3)
        print("Testing all images in class 3...")
        loss3,acc3 = evaluate_all_images_two_inputs(model,three_images,three_images2,three_labels,test_size)
    if cat_num == 5:
        four_images = health_dict[4]
        four_images2 = health_dict2[4]
        num_four = len(four_images)
        four_labels = np.ones(len(four_images))
        four_labels = np.multiply(four_labels,4)
        print("Testing all images in class 4...")
        loss4,acc4 = evaluate_all_images_two_inputs(model,four_images,four_images2,four_labels,test_size)

    num_total = num_zero + num_one + num_two + num_three + num_four

    percent_zero = num_zero/num_total
    loss0_partial = loss0*percent_zero
    acc0_partial = acc0 * percent_zero

    percent_one = num_one/num_total
    loss1_partial = loss1*percent_one
    acc1_partial = acc1 * percent_one

    if cat_num>2:
        pecent_two = num_two/num_total
        loss2_partial = loss2 * pecent_two
        acc2_partial = acc2 * pecent_two
        
        percent_three = num_three/num_total
        loss3_partial = loss3 * percent_three
        acc3_partial = acc3 * percent_three
    if cat_num == 5:
        percent_four = num_four/num_total
        loss4_partial = loss4 * percent_four
        acc4_partial = acc4 * percent_four

    total_loss = loss0_partial + loss1_partial + loss2_partial + loss3_partial + loss4_partial
    total_acc = acc0_partial + acc1_partial + acc2_partial + acc3_partial + acc4_partial
    
    return total_loss,total_acc,loss0,acc0,loss1,acc1,loss2,acc2,loss3,acc3,loss4,acc4




def dual_inception_v3_ensemble_backup(new_image_width, new_image_height,binary_model,non_zero_model,is_trainable=True,learning_rate=0.0001):
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    input_shape = (new_image_width,new_image_height,3)
    image_input = Input(shape = input_shape)
    #sets the inception_v3 model to not update its weights
    non_zero_model.trainable = False
    binary_model.trainable = False
    # non_zero_model.trainable = True
    # binary_model.trainable = True
    non_zero_model = non_zero_model(image_input)
    binary_model = binary_model(image_input)

    output1 = concatenate([binary_model,non_zero_model])
    final_output = layers.PReLU()(output1)
    final_output = layers.Dropout(0.5)(final_output)
    final_output = layers.Dense(5)(final_output)
    final_output = layers.PReLU()(final_output)
    final_output = layers.Dropout(0.5)(final_output)
    final_output = layers.Dense(5)(final_output)
    final_output = layers.PReLU()(final_output)
    final_output = layers.Dropout(0.5)(final_output)
    final_output = layers.Dense(5,activation="softmax")(final_output)

    final_model = models.Model(inputs=[image_input], outputs=final_output)



    radam = tfa.optimizers.RectifiedAdam(lr=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    final_model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    final_model.summary()

    print("Learning rate is: " + str(learning_rate))
    time.sleep(2)
    # quit()
    return final_model



def dual_input_aug_train(model,labels,images1,images2,class_weights,generator,batch):
    gc.collect()
    # generator1=generator
    # generator2=generator
    labels,images1,images2 = shuffle(labels,images1,images2)
    images1 = images1[0]
    images2 = images2[0]
    labels = labels[0]
    current_image1 = cv2.imread(images1)        
    current_image1 = cv2.cvtColor(current_image1, cv2.COLOR_BGR2RGB)
    current_image1 = current_image1.reshape((1, ) + current_image1.shape)
    current_image2 = cv2.imread(images2)        
    current_image2 = cv2.cvtColor(current_image2, cv2.COLOR_BGR2RGB)
    current_image2 = current_image2.reshape((1, ) + current_image2.shape)
    rand_seed = random.randint(0,99999999)
    aug_image1 = generator.flow(current_image1,batch_size = batch,seed=rand_seed)
    aug_image2 = generator.flow(current_image2,batch_size = batch,seed=rand_seed)
    final_labels = np.ones(batch)
    final_labels = np.multiply(final_labels,labels)
    # final_input1 = []
    # final_input1 = np.asarray(final_input1)
    # final_input2 = []
    # final_input2 = np.asarray(final_input2)
    final_input1 = np.empty(shape=[0,256,256,3])
    final_input2 = np.empty(shape=[0,256,256,3])

    for i in range(batch):
        test1 = aug_image1.next()
        test2 = aug_image2.next()

        final_input1 = np.append(final_input1,test1,axis=0)
        final_input2 = np.append(final_input2,test2,axis=0)
        # test1 = np.squeeze(test1,axis=0)
        # test2 = np.squeeze(test2,axis=0)
        # plt.imshow(test1)
        # # plt.plot(test2)
        # plt.show()
        # plt.imshow(test2)
        # plt.show()
    model.train_on_batch([final_input1,final_input2],final_labels,class_weight=class_weights)
    return model
