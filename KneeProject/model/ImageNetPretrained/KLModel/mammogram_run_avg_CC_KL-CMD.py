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

from mammogram_resnet34_KL import *
from mammogram_cross_validation_functions_CMD_KL import *


tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
#tf.app.flags.DEFINE_float('drop_rate', 0.0, 'Dropout rate when training.')
#tf.app.flags.DEFINE_boolean('hidden_layer', False, 'Whether or not to have 1 hidden layer before classifier')
#tf.app.flags.DEFINE_string('optimizer', 'SGD', 'Name of the optimization algorithm we use')
tf.app.flags.DEFINE_string('pooling', 'avg', 'avg or avg pooling')
tf.app.flags.DEFINE_integer('GPU_num', 1, 'Which GPU is used')
tf.app.flags.DEFINE_string('view', 'CC', 'Which breast view weights do I extract: CC or MLO')
tf.app.flags.DEFINE_integer('epochs', 100, 'Number of Epochs')
tf.app.flags.DEFINE_integer('patience', 5, 'Patience for Early Stopping')

FLAGS = tf.app.flags.FLAGS


# Main function for fine tuning models
def main(argv=None):
    
    # Import hdf5 file for Krzysztof's weights
    #filename = 'mammogram/weightsKrzystof.hdf5'
    filename = 'weightsKrzystof.hdf5'
    geras_weights = h5py.File(filename, 'r')

    # Import raw data sets, both have images and labels
    data = np.load('data/KL/0.npz')
    images_cross_validation_all = data['y']
    labels_cross_validation_all = data['x']

    for i in range(1, 50): # Files 0 to 49 are for the baseline images
        data = np.load('data/KL/' + str(i) + '.npz')
        images_cross_validation_all = np.concatenate((images_cross_validation_all, data['y']), axis = 0)
        labels_cross_validation_all = np.concatenate((labels_cross_validation_all, data['x']), axis = 0)

    print(len(labels_cross_validation_all))
    labels_cross_validation_all = labels_cross_validation_all[np.where(labels_cross_validation_all[:,3] == '00')] ## REMOVE THIS LINE IF YOU DON'T WANT JUST BASELINE IMAGES
    images_cross_validation_all = images_cross_validation_all[np.where(labels_cross_validation_all[:,3] == '00')]
    print(len(labels_cross_validation_all))
    print('Images Loaded')

    # Train network with cross validation and corresponding hyperparameters, which are inputed with tf flags
    cross_validation(num_of_folds = 4, GPU_num = 0, 
                     pooling = 'avg', learning_rate = 0.00001, view = 'CC', 
                     patience = 100, epochs = 100, 
                     data = images_cross_validation_all, label_data_all = labels_cross_validation_all, 
                     file_path = 'model_weights/mammogram/all_data/KL_predictions/', weights = geras_weights) 

    cross_validation(num_of_folds = 4, GPU_num = 0, 
                     pooling = 'avg', learning_rate = 0.0001, view = 'CC', 
                     patience = 100, epochs = 100, 
                     data = images_cross_validation_all, label_data_all = labels_cross_validation_all, 
                     file_path = 'model_weights/mammogram/all_data/KL_predictions/', weights = geras_weights) 

    '''cross_validation(num_of_folds = 4, GPU_num = 0, 
                     pooling = 'avg', learning_rate = 0.001, view = 'MLO', 
                     patience = 5, epochs = 100, 
                     data = images_cross_validation_all, label_data_all = labels_cross_validation_all, 
                     file_path = 'model_weights/mammogram/all_data/', weights = geras_weights) 

    cross_validation(num_of_folds = 4, GPU_num = 0, 
                     pooling = 'avg', learning_rate = 0.01, view = 'MLO', 
                     patience = 5, epochs = 100, 
                     data = images_cross_validation_all, label_data_all = labels_cross_validation_all,  
                     file_path = 'model_weights/mammogram/all_data/', weights = geras_weights) 

    cross_validation(num_of_folds = 4, GPU_num = 0, 
                     pooling = 'avg', learning_rate = 0.1, view = 'MLO', 
                     patience = 5, epochs = 100, 
                     data = images_cross_validation_all, label_data_all = labels_cross_validation_all, 
                     file_path = 'model_weights/mammogram/all_data/', weights = geras_weights) 
    '''



if __name__ == "__main__":
    tf.app.run()