import os
import warnings
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import tensorflow as tf
import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
from keras.engine import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from mammogram_resnet34_KL import *

cropped_size = 896
'''
    Func: 
        For each image, subtract mean from each image then divide by sd from each image
    Params: 
        data = array of (reshaped) images
    Returns:
        new_data = preprocessed images, with 0 mean and 1 var
'''
def normalize_with_per_image_average_and_sd(data):
    
    new_data = np.empty([data.shape[0], data.shape[1], data.shape[2]], dtype = np.float64)
    
    for i in range(data.shape[0]):
        new_data[i,:,:]  = data[i,:,:] - np.mean(data[i,:,:])
        new_data[i,:,:] /= np.std(data[i,:,:])
    
    return new_data


def crop_generator(batches, crop_length):
  while True:
    img_height=1024
    img_width=1024
    batch_x, batch_y = next(batches)
    start_y = (img_height - crop_length) // 2
    start_x = (img_width - crop_length) // 2
    if K.image_data_format() == 'channels_last':
        batch_crops = batch_x[:, start_x:(img_width - start_x), start_y:(img_height - start_y), :]
    else:
        batch_crops = batch_x[:, :, start_x:(img_width - start_x), start_y:(img_height - start_y)]
    yield (batch_crops, batch_y)


'''
    Func: 
    	Code to train model
    Params: 
    	model = model created in above function
        patience = num of epochs for early stopping
		batch_size = num of training examples per batch -- need to adjust due to memory issues
		train_images = training images
		train_labels = training labels
		val_images = validation images
		val_labels = validation labels
		path = file path to save stuff
		epochs = number of epochs to run
'''
def train_model(model, patience, batch_size, train_images, train_labels, val_images, val_labels, path, epochs):
    
    # Early Stopping callback that can be found on Keras website
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience)
    
    # Create path to save weights with model checkpoint
    weights_path = path + 'weights-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.hdf5' #'best_weights.hdf5'
    model_checkpoint = ModelCheckpoint(weights_path, monitor = 'val_loss', save_best_only = False, save_weights_only=True,
                                       verbose=0, period=1)
    model_checkpoint_best = ModelCheckpoint(path+'best_weights.hdf5', monitor = 'val_loss', save_best_only = True, save_weights_only=True,
                                            verbose=0, period=1)
    # Save loss and accuracy curves using Tensorboard
    tensorboard_callback = TensorBoard(log_dir = path, 
                                       histogram_freq = 0, 
                                       write_graph = False, 
                                       write_grads = False, 
                                       write_images = False,
                                       batch_size = batch_size)
    
    # Additional callback if we anneal the learning rate

    callbacks_list = [early_stopping, model_checkpoint, model_checkpoint_best, tensorboard_callback]

    train_datagen = ImageDataGenerator(horizontal_flip = True,samplewise_center=True)
    train_generator = train_datagen.flow(train_images, train_labels, batch_size = batch_size)

    train_crops = crop_generator(train_generator, cropped_size)

    # Train the model 
    model.fit_generator(train_crops, epochs=epochs, steps_per_epoch = train_labels.shape[0] // batch_size,
                        callbacks = callbacks_list,
                        validation_data = (val_images, val_labels))

    final_path = path + 'final_weights.hdf5'
    model.save_weights(final_path)  # )
    

'''
    Func: 
    	Code to run stratified cross validation to train my network
    Params: 
    	num_of_folds = number of folds to cross validate
        GPU_num = the number of the GPU we run on
		pooling = 'avg' or 'max'
		learning_rate = learning rate
		view = 'CC' or 'MLO'
		patience = patience for early stopping
		epochs = num of epochs to train
		data = original 1024x1024 images -- preprocessing is done within cross validation function
		labels = labels corresponding to images
		file_path = path to save network weights, curves, and tensorboard callbacks
		weights = hdf5 object for weights
'''

def cross_validation(num_of_folds, 
                     GPU_num, 
                     pooling,
                     learning_rate,  
                     view, 
                     patience,
                     epochs,
                     data, 
                     label_data_all, 
                     file_path,
                     weights):

    # Set seed for k-fold cross validation and then generate folds
    seed = 1234
    skf = StratifiedKFold(n_splits = num_of_folds, shuffle=True, random_state=seed)
    
    # Generate specific folder path for saving model weights with corresponding hyperparameters
    model_path = file_path + 'view%s_pooling%s_lr%s_gpu%s/' % (view, pooling, learning_rate, GPU_num)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    preprocessed_images = normalize_with_per_image_average_and_sd(data)
    preprocessed_images = preprocessed_images.reshape((preprocessed_images.shape[0], preprocessed_images.shape[1], preprocessed_images.shape[2], 1))
    
    labels = label_data_all[:,1]
    #print(labels)
    labels = labels.astype(int)
    # Training with cross validation 
    i = 1
    for train_index, test_index in skf.split(np.zeros(labels.shape[0]), labels):         
        model = ResNet34(input_shape=(cropped_size,cropped_size,1),
             			 pooling=pooling, 
             			 bn_axis=3, 
             			 hidden_layer=False, 
             			 optimizer='ADAM', 
             			 learning_rate=learning_rate)

        model = set_weights(model=model, view=view, weights=weights)

        # Track fold number
        print('Running Fold', i, '/', num_of_folds)
            
        #if i == 1:
        #  model.summary()
          
        # Generate folder for each fold to store model weights
        fold_path = model_path + 'Fold_' + str(i) + '/'
        #print(fold_path)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)    
        
        # Train model
        train_model(model=model, patience = patience, batch_size = 8,
                    train_images = preprocessed_images[train_index], train_labels = np_utils.to_categorical(labels[train_index]),
                    val_images = preprocessed_images[test_index,64:1024-64,64:1024-64], val_labels = np_utils.to_categorical(labels[test_index]),
                    path = fold_path, epochs=epochs)

        inference(pooling=pooling, learning_rate=learning_rate, weights_path=fold_path+'best_weights.hdf5', 
                  view=view, test_data=preprocessed_images[test_index,64:1024-64,64:1024-64], labels=np_utils.to_categorical(label_data_all[test_index]), prediction_path=fold_path)

        del model
        i += 1


def inference(pooling, learning_rate, weights_path, view, test_data, labels,prediction_path):
    model = ResNet34(input_shape=(cropped_size,cropped_size,1),
                      pooling=pooling, 
                      bn_axis=3, 
                      hidden_layer=False, 
                      optimizer='ADAM', 
                      learning_rate=learning_rate)
    model.load_weights(weights_path)
    print('Model weights are loaded')

    #test_data = normalize_with_per_image_average_and_sd(test_data) -- data is already preprocessed, don't need to do it again
    predictions = model.predict(test_data, batch_size = 8)  
    print('Predictions are complete')

    prediction_file = prediction_path + view + '_test_predictions.csv'  
    print(prediction_file)

    # Array of results: col 1 = index of example, col 2 = predicted probability, col 3 = ground truth label
    return_array = np.empty(shape=(predictions.shape[0], 4))
    return_array[:,0] = labels[:,0]
    return_array[:,1] = predictions[:,0]
    return_array[:,2] = labels[:,1]
    return_array[:,3] = labels[:,2]

    np.savetxt(prediction_file, return_array, delimiter=',')