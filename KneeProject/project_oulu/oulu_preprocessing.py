from xray_processor import process_file, image_preprocessing_oulu
import os
import numpy as np
import argparse
import pandas as pd
import time
import os
import random

if __name__ == "__main__":
    summaryFile = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/Dataset/OAI_summary.csv'
    summary = pd.read_csv(summaryFile)
    print(summary)
    save_dir = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_oulu_large'
    count = 0
    oaiDataset ='/gpfs/data/denizlab/Datasets/OAI_original'
    subject_id = summary.ID.unique().tolist()
    random.shuffle(subject_id)
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2
    num_subject = len(subject_id)
    train_subject_ids = subject_id[: int(train_size * num_subject)]
    subject_id = subject_id[int(train_size * num_subject):]
    val_subject_ids = subject_id[:int(val_size * num_subject)]
    subject_id = subject_id[int(val_size * num_subject):]
    test_subject_ids = subject_id
    print('Training data {}; Val data {}; Test data {}'.\
          format(len(train_subject_ids),len(val_subject_ids),len(test_subject_ids)))
    count = 0
    for idx in train_subject_ids:
        train_save_dir = os.path.join(save_dir, 'train')
        kl_grades = summary.loc[summary.ID == idx]
        for data_dir in kl_grades.Folder.unique().tolist():
            subject_info = kl_grades.loc[kl_grades.Folder == data_dir]
            month = subject_info.Visit.tolist()[0]
            subject = subject_info.ID.tolist()[0]
            data_dir = os.path.join(oaiDataset, month, data_dir)
            bbox = image_preprocessing_oulu(data_dir, '001')
            bbox = bbox.split(' ')
            bbox = np.array([int(i) for i in bbox[1:]])
            fname = '{}_{}'.format(month, subject)
            print(fname)
            if -1 in bbox:
                continue
            count += 1
            try:
                left_kl = int(kl_grades.loc[kl_grades.SIDE == 2].KLG.tolist()[0])
                right_kl = int(kl_grades.loc[kl_grades.SIDE == 1].KLG.tolist()[0])
            except Exception as e:
                print(e)
                continue
            # ATTENTION: the KL grades specified below are FAKE and you need to
            #            retrieve the correct values by yourself (e.g. extract
            #            from the filenames or pass from metadata dataframe)

            process_file(idx, '001', fname, data_dir, train_save_dir, bbox, left_kl, right_kl)
            count += 2
    for idx in val_subject_ids:
        val_save_dir = os.path.join(save_dir,'val')
        kl_grades = summary.loc[summary.ID == idx]
        for data_dir in kl_grades.Folder.unique().tolist():
            subject_info = kl_grades.loc[kl_grades.Folder == data_dir]
            month = subject_info.Visit.tolist()[0]
            subject = subject_info.ID.tolist()[0]
            data_dir = os.path.join(oaiDataset, month, data_dir)
            bbox = image_preprocessing_oulu(data_dir, '001')
            bbox = bbox.split(' ')
            bbox = np.array([int(i) for i in bbox[1:]])
            fname = '{}_{}'.format(month, subject)
            print(fname)
            if -1 in bbox:
                continue
            count += 1
            try:
                left_kl = int(kl_grades.loc[kl_grades.SIDE == 2].KLG.tolist()[0])
                right_kl = int(kl_grades.loc[kl_grades.SIDE == 1].KLG.tolist()[0])
            except Exception as e:
                print(e)
                continue
            # ATTENTION: the KL grades specified below are FAKE and you need to
            #            retrieve the correct values by yourself (e.g. extract
            #            from the filenames or pass from metadata dataframe)

            process_file(idx, '001', fname, data_dir, val_save_dir, bbox, left_kl, right_kl)
            count += 2

    for idx in test_subject_ids:
        test_save_dir = os.path.join(save_dir,'test')
        kl_grades = summary.loc[summary.ID == idx]
        for data_dir in kl_grades.Folder.unique().tolist():
            subject_info = kl_grades.loc[kl_grades.Folder == data_dir]
            month = subject_info.Visit.tolist()[0]
            subject = subject_info.ID.tolist()[0]
            data_dir = os.path.join(oaiDataset, month, data_dir)
            bbox = image_preprocessing_oulu(data_dir, '001')
            bbox = bbox.split(' ')
            bbox = np.array([int(i) for i in bbox[1:]])
            fname = '{}_{}'.format(month, subject)
            print(fname)
            if -1 in bbox:
                continue
            count += 1
            try:
                left_kl = int(kl_grades.loc[kl_grades.SIDE == 2].KLG.tolist()[0])
                right_kl = int(kl_grades.loc[kl_grades.SIDE == 1].KLG.tolist()[0])
            except Exception as e:
                print(e)
                continue
            # ATTENTION: the KL grades specified below are FAKE and you need to
            #            retrieve the correct values by yourself (e.g. extract
            #            from the filenames or pass from metadata dataframe)

            process_file(idx, '001', fname, data_dir, test_save_dir, bbox, left_kl, right_kl)
            count += 2
    print('{} / {} samples are detected'.format(count, len(summary.Folder.unique().tolist())))