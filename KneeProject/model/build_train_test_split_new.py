import pandas as pd
import numpy as np
import os
import sys
import time
import random
random.seed(0)
def build_train_test_split(summary_path):
    '''

    :param summary_path:
    :return:
    '''
    # read summary into df
    df = pd.read_csv(summary_path + 'OAI_summary.csv')
    print(df.shape)
    df = df.dropna()
    print(df.shape)
    # split train test based on patient ID
    participant_ids = df['ID'].unique().tolist()
    train_size = int(len(participant_ids) * 0.7)
    test_size = int(len(participant_ids) * 0.2)
    # sample patient ID not KL grade
    train_ids = random.sample(k=train_size, population=participant_ids)
    participant_ids = [i for i in participant_ids if i not in train_ids]
    test_ids = random.sample(k=test_size, population=participant_ids)
    val_ids = [i for i in participant_ids if i not in test_ids]
    print('Number of participants: Train {}; Val {}; Test {}.'\
          .format(len(train_ids),len(val_ids),len(test_ids)))
    train = df.loc[df['ID'].isin(train_ids)]
    test = df.loc[df['ID'].isin(test_ids)]
    val = df.loc[df['ID'].isin(val_ids)]

    print('Training set {}, validation set {},test set {}'.format(train.shape[0], val.shape[0],test.shape[0]))
    train.to_csv(summary_path + 'train.csv',index = False)
    val.to_csv(summary_path + 'val.csv',index = False)
    test.to_csv(summary_path + 'test.csv',index = False)

if __name__ == '__main__':
    summary_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/Dataset/'
    build_train_test_split(summary_path=summary_path) # split train test
