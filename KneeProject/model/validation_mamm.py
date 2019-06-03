from KLModel.mammogram_resnet34_KL import *
import pandas as pd
import numpy as np
import os
import sys
import time
from KLModel.image_data_generator import *
import tensorflow as tf
from keras.models import load_model
if __name__ == '__main__':
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    weights_path = '/gpfs/data/denizlab/Users/bz1030/model/model_weights/mammogram/fine_tuning_last_0'
    print('Load model ...')
    resnet34 = load_model(weights_path + '/resnet34_from_mamm_randCrop_train_checkpoint_60000.h5')
    print('Load test data summary ...')
    test = pd.read_csv(summary_path + 'test.csv')
    print('Test set {}'.format(test.shape[0]))
    image_generator_eval(test,resnet34,HOME_PATH,batch_size = 128,random_crop_img = True)