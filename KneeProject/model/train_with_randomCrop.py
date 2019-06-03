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
from KLModel.mammogram_resnet34_KL import *
from KLModel.image_data_generator import *
from keras.models import load_model
if __name__ == '__main__':
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    loadWeights = None #'resnet34_from_scratch_randCrop_checkpoint_0.h5'
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    model_path = '/gpfs/data/denizlab/Users/bz1030/model/model_weights/from_scratch/train_from_scratch_randC3/'
    if not os.path.exists((model_path)):
        os.makedirs(model_path)


    if loadWeights:
        print('Load Model ...')
        resnet34 = load_model(model_path + loadWeights)
    else:
        print('Create Model ...')
        resnet34 = ResNet34(input_shape=(896, 896, 1))
    train = pd.read_csv(summary_path + 'train.csv')
    val = pd.read_csv(summary_path + 'val.csv')  # split train - test set.
    # setup batch size
    train_batch_size = 8
    val_batch_size = 128
    # create data loader
    train_dataset = image_generator(batch_size=train_batch_size, home_path=HOME_PATH, summary=train,random_crop_img = True)
    print('Start Training ...')
    iteration = 200000
    for it in range(1,iteration + 1):
        X, y = next(train_dataset)
        train_loss, train_acc = resnet34.train_on_batch(X, y)
        print('Iteration {};Train loss {};Train accuracy: {}.'.format(it,train_loss, train_acc))
        if it > 60000 and it % 10000 == 0:
            val_acc = image_generator_eval(val, resnet34, HOME_PATH, batch_size=128, random_crop_img=True)
            print('Validation accuracy: {}.'.format(val_acc))
            resnet34.save(model_path + 'resnet34_from_scratch_randCrop_checkpoint_{}.h5' \
                          .format(it))
        elif it % 10000 == 0:
            resnet34.save(model_path + 'resnet34_from_scratch_randCrop_checkpoint_{}.h5' \
                          .format(it))



