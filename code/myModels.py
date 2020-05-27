import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout,Input,concatenate
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import numpy as np

import time

from preprocessData import *
from utils import *
# import efficientnet.keras as efn 
# from keras_efficientnets import EfficientNetB5
#import efficientnet.keras as efn 
from efficientnet.tfkeras import EfficientNetB7 as Net

import tensorflow_addons as tfa
import gc


from tensorflow.keras.utils import plot_model





# from tensorflow.keras.applications import inception_v3

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

def transfer_learning_model_inception_v3(new_image_width, new_image_height,is_trainable=True):
    if is_trainable is None:
        is_trainable = True
    #loads the inception_v3 model, removes the last layer, and sets inputs to the size needed
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False,input_shape=(new_image_width, new_image_height, 3))
    # base_model = EfficientNetB5(weights='imagenet',include_top=False,input_shape=(new_image_width, new_image_height, 3))

    #sets the inception_v3 model to not update its weights
    base_model.trainable = is_trainable
    #layer to convert the features to a single n-elemnt vector per image
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    #base_model.summary()
    #adds the two layers for transfer learning
    model = models.Sequential([base_model,global_average_layer])
    #adds dense and dropout layers for final output
    model.add(layers.Dense(1024))
    model.add(layers.PReLU())
    model.add(Dropout(0.5))
    model.add(layers.Dense(512))
    model.add(layers.PReLU())
    model.add(Dropout(0.5))
    model.add(layers.Dense(256))
    model.add(layers.PReLU())
    model.add(Dropout(0.5))

    model.add(layers.Dense(5, activation='softmax'))

    radam = tfa.optimizers.RectifiedAdam(lr=0.00001)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    # model.compile(optimizer=Adam(lr=0.00001),
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model
    model.compile(optimizer=ranger,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    
    return model

def inception_v3_multiple_inputs(image_width,image_height):
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
    final_model.summary()
    radam = tfa.optimizers.RectifiedAdam(lr=0.000001)
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

    test_size = 100
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






