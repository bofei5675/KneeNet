from xray_processor import process_file
import os
import numpy as np
import argparse
from detector import KneeLocalizer,worker
import pandas as pd
def image_preprocessing_oulu(data_folder, file):
    localizer = KneeLocalizer()

    bbox = worker(file, data_folder, localizer)  # output a string
    return bbox
import time
import os
if __name__ == "__main__":

    summaryFile = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/Dataset/OAI_summary.csv'
    summary = pd.read_csv(summaryFile)
    print(summary)
    save_dir = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_oulu'
    oaiDataset ='/gpfs/data/denizlab/Datasets/OAI_original'
    for idx,data_dir in enumerate(summary.Folder.unique().tolist()):

        kl_grades = summary.loc[summary.Folder == data_dir]
        month = kl_grades.Visit.tolist()[0]
        subject = kl_grades.ID.tolist()[0]
        data_dir = os.path.join(oaiDataset,month,data_dir)
        bbox = image_preprocessing_oulu(data_dir, '001')
        bbox = bbox.split(' ')
        bbox = np.array([int(i) for i in bbox[1:]])
        print(bbox)
        fname = '{}_{}'.format(month, subject)
        if -1 in bbox:
            continue
        left_kl = int(kl_grades.loc[kl_grades.SIDE ==2].KLG.tolist()[0])
        right_kl = int(kl_grades.loc[kl_grades.SIDE ==1].KLG.tolist()[0])
        # ATTENTION: the KL grades specified below are FAKE and you need to
        #            retrieve the correct values by yourself (e.g. extract
        #            from the filenames or pass from metadata dataframe)

        process_file(idx, fname, data_dir, save_dir, bbox, left_kl, right_kl)