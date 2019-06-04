import os
import numpy as np
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import h5py
import pandas as pd
import time
import os
import numpy as np
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import random as rand
import time
import scipy.ndimage as ndimage
from utils import *

def get_KL_grade(file_path,month_file):
    '''
    Read KL Grade as a data frame
    :param file_path: e.g. '/gpfs/data/denizlab/Datasets/OAI/ClinicalFromNDA/X-Ray Image Assessments_SAS/Semi-Quant Scoring_SAS/kxr_sq_bu00.txt'
    :return:
    '''
    df = pd.read_csv(file_path,sep='|')
    df.columns = [col.upper() for col in df.columns] # some files READPRJ is not captilized(???)
    print('####### Obtaining KL Grade File ###############')
    df= df[['ID','SIDE','V{}XRKL'.format(month_file),'READPRJ']]
    df = df.loc[df['READPRJ'] == 15] # this is a project code from the experiment
    return df
def get_KL(df,patientID,side,month_file):
    '''
    Get KLG from dataframe
    :param df:
    :param patientID:
    :param side: 1:right, 2:left
    :return:
    '''
    patientInfo = df.loc[df['ID'] == patientID]
    kl_grade = patientInfo.loc[patientInfo['SIDE'] == side,'V{}XRKL'.format(month_file)]
    if kl_grade.shape[0] == 0:
        kl_grade ='NA'
    return np.squeeze(kl_grade)

def read_dicome_and_process(content_file_path='/gpfs/data/denizlab/Datasets/OAI_original/',month = '00m',method = 'mean',
                            save_dir = '/gpfs/data/denizlab/Users/bz1030/test/test1/'):
    '''
    Read the content files and process all DICOM Image

    :param content_file_path:
    :param month:
    :return:
    '''
    if method not in ['mean','svm','mix']:
        raise ValueError('Please use method of mean, svm, or mix')
    monthToKL = {
        '00m':'00',
        '12m': '01',
        '18m': '02',
        '24m': '03',
        '30m': '04',
        '36m': '05',
        '48m': '06',
        '72m': '08',
        '96m': '10',
    } # this map is obtained from Cem and README file of dataset.
    klGradeFilePath = '/gpfs/data/denizlab/Datasets/OAI/ClinicalFromNDA/X-Ray Image Assessments_SAS/Semi-Quant Scoring_SAS/'
    klGradeFileName = 'kxr_sq_bu{}.txt'.format(monthToKL[month])
    klGradeFilePath = os.path.join(klGradeFilePath,klGradeFileName)
    content_file_path = os.path.join(content_file_path,month)
    file_name = 'contents.csv'
    count = 0
    KL_Grade = get_KL_grade(klGradeFilePath,monthToKL[month])
    summary = {
        'File Name':[],
        'Folder':[],
        'Participant ID':[],
        'Study Date':[],
        'Bar Code':[],
        'Description':[],
        'Image Size':[],
        'KLG':[],
        'Method':[],
        'IsSuccessful':[]
    }
    svm_not_found = 0
    with open(os.path.join(content_file_path,file_name),'r') as f:
        next(f) # skip first row
        for line in f:
            line = line.rstrip().replace('"','').split(',') # split each line by csv
            data_path,patientID,studyDate,barCode,description = line[0],line[1],line[2],line[3],line[4]
            description = description.rstrip().replace('"','').replace(' ','').split('^') # split fields inside description
            # only look at XRAY and Knee data
            if description[1] == 'XRAY' and description[-1] == 'KNEE':
                data_path = content_file_path + '/'+ data_path.replace('"','')
                data_files = os.listdir(data_path)
                for data_file in data_files:
                    print(data_path,data_files)
                    img, data, img_before = image_preprocessing(os.path.join(data_path, data_file))
                    left_svm, right_svm = image_preprocessing_oulu(data_path,data_file)
                    if left_svm is None:
                        svm_not_found += 1
                    left,right = extract_knee(img,0), extract_knee(img,1) # left is 0, and right is 1
                    left_kl,right_kl = get_KL(KL_Grade,int(patientID),2,monthToKL[month]),get_KL(KL_Grade,int(patientID),1,monthToKL[month])
                    # create hdf5 file
                    if method == 'mean':
                        create_hdf5_file(summary,left,data,patientID,studyDate,barCode,'LEFT',left_kl,
                                         month,data_path,save_dir=save_dir,method = method)
                        create_hdf5_file(summary,right,data,patientID,studyDate,barCode,'RIGHT',right_kl,
                                         month,data_path,save_dir=save_dir,method = method)
                    elif method == 'svm':
                        create_hdf5_file(summary, left_svm, data, patientID, studyDate, barCode, 'LEFT', left_kl, month,
                                         data_path,save_dir=save_dir, method=method)
                        create_hdf5_file(summary, right_svm, data, patientID, studyDate, barCode, 'RIGHT', right_kl, month,
                                         data_path, save_dir=save_dir,method=method)
                    elif method == 'mix':
                        if left_svm is not None:
                            create_hdf5_file(summary, left_svm, data, patientID, studyDate, barCode, 'LEFT', left_kl, month,
                                             data_path,save_dir=save_dir, method='mix',isSuccessful=1)
                            create_hdf5_file(summary, right_svm, data, patientID, studyDate, barCode, 'RIGHT', right_kl,
                                             month,
                                             data_path, save_dir=save_dir,method='mix',isSuccessful=1)
                        else:
                            create_hdf5_file(summary, left, data, patientID, studyDate, barCode, 'LEFT', left_kl, month,
                                             data_path, save_dir=save_dir,method='mix',isSuccessful=0)
                            create_hdf5_file(summary, right, data, patientID, studyDate, barCode, 'RIGHT', right_kl,
                                             month, data_path, save_dir=save_dir,method='mix',isSuccessful=0)


                    generate_figure(img_before,
                                    img,
                                    left,right,left_svm,right_svm,
                                    '../test/test_image/{}/'.format(month),'{}_{}_{}.png'.format(patientID,studyDate,barCode))
                    count += 1

                    if count % 100 == 0:
                        df = pd.DataFrame(summary)
                        df.to_csv(save_dir + '/' +'summary_{}.csv'.format(month), index=False)
    print('Total processed:',count)
    print('Knee not found by SVM:',svm_not_found)
    df = pd.DataFrame(summary)
    df.to_csv(save_dir + '/' + 'summary_{}.csv'.format(month),index = False)

def create_hdf5_file(summary,image, data,patientID, studyDate, barCode,description,kl_grade,month,data_path,
                     save_dir = '/gpfs/data/denizlab/Users/bz1030/test/test1/',method = 'mean',isSuccessful=1):
    '''
    Save the HDF5 file to the directory
    :param image: image array 1024 x 1024
    :param data:
    :param patientID:
    :param studyDate:
    :param barCode:
    :param description:
    :param file_name:
    :param save_dir:
    :return:
    '''
    save_dir = os.path.join(save_dir,method,str(month))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Save to {}'.format(save_dir))
    file_name = str(patientID) +'_'+ month + '_'+ description +'_' + 'KNEE.hdf5'
    pixelDimensions = image.shape
    pixelSpacing = '%.3fx%.3f' % (float(data.PixelSpacing[0]),float(data.PixelSpacing[1]))
    if image is not None and pixelDimensions[0] == 1024 and pixelDimensions[1] == 1024:
        # modify summary dictionary
        summary['File Name'].append(file_name)
        summary['Folder'].append(data_path)
        summary['Participant ID'].append(patientID)
        summary['Study Date'].append(studyDate)
        summary['Bar Code'].append(barCode)
        summary['Description'].append(description)
        summary['Image Size'].append('{}x{}'.format(*pixelDimensions))
        summary['KLG'].append(kl_grade)
        summary['Method'].append(method)
        summary['IsSuccessful'].append(isSuccessful)
        # create hdf5 file
        f = h5py.File(save_dir +'/' + file_name,'w')
        f.create_dataset('data', data = image)
        f.create_dataset('PixelDims',data = pixelDimensions)
        f.create_dataset('PixelSpacing',data= pixelSpacing)
        f.create_dataset('Folder',data=save_dir +'/' + file_name)
        f.close()
    '''
    else:
        # modify summary dictionary
        summary['File Name'].append(file_name)
        summary['Folder'].append(data_path)
        summary['Participant ID'].append(patientID)
        summary['Study Date'].append(studyDate)
        summary['Bar Code'].append(barCode)
        summary['Description'].append(description)
        summary['Image Size'].append('{}x{}'.format(*pixelDimensions))
        summary['KLG'].append(kl_grade)
        summary['Method'].append(method)
        summary['IsSuccessful'].append(isSuccessful)
    '''

def generate_figure(img_array_before,img_array_after,left,right,
                    left_SVM = None, right_SVM = None,save_dir=None,file_name=None):
    '''
    :param img_array_before:
    :param img_array_after:
    :param left:
    :param right:
    :param left_SVM:
    :param right_SVM:
    :param save_dir:
    :param file_name:
    :return:
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) # make sure the directory is created.
    rows = 2 if left_SVM is None else 3
    cols = 2
    f, ax = plt.subplots(rows,cols,dpi=300)

    ax[0,0].imshow(img_array_before)
    ax[0,1].imshow(img_array_after)
    ax[1,0].imshow(left)
    ax[1,1].imshow(right)
    ax[0,0].set_title('Before preprocessing')
    ax[0,1].set_title('After preprocessing')
    ax[1,0].set_title('Left')
    ax[1,1].set_title('Right')
    if rows == 3:
        ax[2, 0].imshow(left_SVM)
        ax[2, 1].imshow(right_SVM)
        ax[2, 0].set_title('Left_Knee_OULU')
        ax[2, 1].set_title('Right_Knee_OULU')
    f.tight_layout()
    f.savefig(os.path.join(save_dir,file_name),dpi=300,bbox_inches='tight')
    plt.close('all')
