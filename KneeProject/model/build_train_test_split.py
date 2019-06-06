#from KLModel.mammogram_resnet34_KL import *
import pandas as pd
import numpy as np
import os
import sys
import time
import random
random.seed(0)
def build_summary_file(summary_path):
    summary_files = os.listdir(summary_path)
    summary_files = [i for i in summary_files if '.csv' in i and 'summary_' in i]
    print(summary_files)
    dfs = []
    for each in summary_files:
        dfs.append(pd.read_csv(summary_path  + each))
    print(len(dfs))
    summary = pd.concat(dfs)
    summary = summary.dropna()
    print(summary.head())
    print(summary.shape)
    summary.to_csv(summary_path + 'summary.csv',index = False)
def build_train_test_split(summary_path):
    # read summary into df
    df = pd.read_csv(summary_path + 'summary.csv')
    # split train test based on patient ID
    participant_ids = df['Participant ID'].unique().tolist()
    train_size = int(len(participant_ids) * 0.7)
    test_size = int(len(participant_ids) * 0.2)
    # sample patient ID not KL grade
    train_ids = random.sample(k=train_size, population=participant_ids)
    participant_ids = [i for i in participant_ids if i not in train_ids]
    test_ids = random.sample(k=test_size, population=participant_ids)
    val_ids = [i for i in participant_ids if i not in test_ids]
    train = df.loc[df['Participant ID'].isin(train_ids)]
    test = df.loc[df['Participant ID'].isin(test_ids)]
    val = df.loc[df['Participant ID'].isin(val_ids)]

    # then samples t
    print('Training set {}, validation set {},test set {}'.format(train.shape[0], val.shape[0],test.shape[0]))
    train.to_csv(summary_path + 'train.csv',index = False)
    val.to_csv(summary_path + 'val.csv',index = False)
    test.to_csv(summary_path + 'test.csv',index = False)

if __name__ == '__main__':
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    build_summary_file(summary_path=summary_path) # generate summary file
    build_train_test_split(summary_path=summary_path) # split train test
