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
import sys
'''
Train the model with mamm weights and fine tuning given layers.
'''
if __name__ == '__main__':
    fine_tune_layers = int(sys.argv[1]) # change the number of layers to fine tune
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    model_path = '/gpfs/data/denizlab/Users/bz1030/model/model_weights/mammogram/fine_tuning_v2_last_{}/'.format(fine_tune_layers)
    if not os.path.exists((model_path)):
        os.makedirs(model_path)

    resnet34 = ResNet34(input_shape=(896, 896, 1))

    weights_file = '/gpfs/data/denizlab/Users/bz1030/model/model_weights/mammogram/weightsKrzystof_ResNet_new_architecture.hdf5'

    geras_weights = h5py.File(weights_file,'r')

    view = 'CC'
    print('Loading parameters ...')
    resnet34 = set_weights(model=resnet34, view=view, weights=geras_weights)
    if fine_tune_layers != 0:
        print('Fine tuning last {} layers'.format(fine_tune_layers))
        for layer in resnet34.layers[:-fine_tune_layers]:
            layer.trainable = False
    print('Finished loading ...')
    train = pd.read_csv(summary_path + 'train.csv')
    val = pd.read_csv(summary_path + 'val.csv')  # split train - test set.
    train_batch_size = 8
    val__batch_size = 128
    train_dataset = image_generator(batch_size=train_batch_size, home_path=HOME_PATH, summary=train, random_crop_img=True)
    print('Start Training ...')
    iteration = 200000
    for it in range(1,iteration):
        X, y = next(train_dataset)
        train_loss, train_acc = resnet34.train_on_batch(X, y)
        print('Iteration{}; Train loss {};Train accuracy: {}.'.format(it,train_loss, train_acc))
        if it % 10000 == 0:
            val_acc = image_generator_eval(val, resnet34, HOME_PATH, batch_size=128, random_crop_img=True)
            print('Validation accuracy: {}.'.format(val_acc))
            resnet34.save(model_path + 'resnet34_from_mamm_randCrop_train_checkpoint_{}.h5' \
                          .format(it))




