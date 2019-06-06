from KLModel.mammogram_resnet34_KL import *
import pandas as pd
import numpy as np
import os
import sys
import time
from KLModel.image_data_generator import *
import tensorflow as tf
'''
Train the model without doing any thing to the raw image.
'''
if __name__ == '__main__':
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    batch_size = 8
    resnet34 = ResNet34(input_shape=(1024, 1024, 1))
    train = pd.read_csv(summary_path + 'train.csv')

    test = pd.read_csv(summary_path + 'test.csv') # split train - test set.

    print('Training set {}, test set {}'.format(train.shape[0], test.shape[0]))
    train_dataset = image_generator(batch_size=batch_size, home_path=HOME_PATH, summary=train)
    test_dataset = image_generator(batch_size=batch_size,home_path = HOME_PATH,summary=test)
    iteration = 200000
    for iter in range(iteration):
        X, y = next(train_dataset)
        train_loss, train_acc = resnet34.train_on_batch(X,y)
        print('Train loss {};Train accuracy: {}.'.format(train_loss, train_acc))
        if iter % 5000 == 0:
            X_test,y_test = next(test_dataset)
            test_loss, test_acc = resnet34.evaluate(X_test,y_test)
            print('Test loss {};Test accuracy: {}.'.format(test_loss,test_acc))
            resnet34.save('resnet34_from_scratch_{}.h5'.format(iter))
