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
from oulukneeloc import SVM_MODEL_PATH
from oulukneeloc.proposals import (read_dicom, get_joint_y_proposals,
                                   preprocess_xray)
from detector import KneeLocalizer,worker
from utils import *

def get_KL_grade(file_path='/gpfs/data/denizlab/Datasets/OAI/ClinicalFromNDA/X-Ray Image Assessments_SAS/Semi-Quant Scoring_SAS/kxr_sq_bu00.txt'):
    '''
    Read KL Grade as a data frame
    :param file_path:
    :return:
    '''
    df = pd.read_csv(file_path,sep='|')[['ID','SIDE','V00XRKL','READPRJ']]
    df = df.loc[df['READPRJ'] == 15]
    return df
def get_KL(df,patientID,side):
    '''
    Get KLG from dataframe
    :param df:
    :param patientID:
    :param side: 1:right, 2:left
    :return:
    '''
    patientInfo = df.loc[df['ID'] == patientID]
    kl_grade = patientInfo.loc[patientInfo['SIDE'] == side,'V00XRKL']
    if kl_grade.shape[0] == 0:
        kl_grade ='NA'
    return np.squeeze(kl_grade)

def read_dicome_and_process(content_file_path='/gpfs/data/denizlab/Datasets/OAI_original/',month = '00m'):
    '''
    Read the content files and process all DICOM Image

    :param content_file_path:
    :param month:
    :return:
    '''
    content_file_path = os.path.join(content_file_path,month)
    file_name = 'contents.csv'
    count = 0
    KL_Grade = get_KL_grade()
    summary = {
        'File Name':[],
        'Folder':[],
        'Participant ID':[],
        'Study Date':[],
        'Bar Code':[],
        'Description':[],
        'Image Size':[],
        'KLG':[]
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
                    img,data,img_before = image_preprocessing(os.path.join(data_path,data_file))
                    left_svm, right_svm = image_preprocessing_oulu(data_path,data_file)
                    if left_svm is None:
                        svm_not_found += 1
                    left,right = extract_knee(img,0), extract_knee(img,1)
                    left_kl,right_kl = get_KL(KL_Grade,int(patientID),2),get_KL(KL_Grade,int(patientID),1)
                    # create hdf5 file
                    create_hdf5_file(summary,left,data,patientID,studyDate,barCode,'LEFT',left_kl,month,data_path)
                    create_hdf5_file(summary,right,data,patientID,studyDate,barCode,'RIGHT',right_kl,month,data_path)
                    generate_figure(img_before,
                                    img,
                                    left,right,left_svm,right_svm,
                                    '../test/test_image/','{}_{}_{}.png'.format(patientID,studyDate,barCode))
                    count += 1
            if count and count >= 20:
                break

    print('Total processed:',count)
    print('Knee not found by SVM:',svm_not_found)
    df = pd.DataFrame(summary)
    df.to_csv('summary.csv',index = False)

def create_hdf5_file(summary,image, data,patientID, studyDate, barCode,description,kl_grade,month,data_path,
                     save_dir = '/gpfs/data/denizlab/Users/bz1030/test/test1/'):
    '''

    :param image:
    :param data:
    :param patientID:
    :param studyDate:
    :param barCode:
    :param description:
    :param file_name:
    :param save_dir:
    :return:
    '''

    file_name = str(patientID) +'_'+ month + '_'+ description +'_' + 'KNEE.hdf5'
    pixelDimensions = image.shape
    pixelSpacing = '%.3fx%.3f' % (float(data.PixelSpacing[0]),float(data.PixelSpacing[1]))

    # modify summary dictionary
    summary['File Name'].append(file_name)
    summary['Folder'].append(data_path)
    summary['Participant ID'].append(patientID)
    summary['Study Date'].append(studyDate)
    summary['Bar Code'].append(barCode)
    summary['Description'].append(description)
    summary['Image Size'].append('{}x{}'.format(*pixelDimensions))
    summary['KLG'].append(kl_grade)
    # create hdf5 file
    f = h5py.File(save_dir + file_name,'w')
    f.create_dataset('data', data = image)
    f.create_dataset('PixelDims',data = pixelDimensions)
    f.create_dataset('PixelSpacing',data= pixelSpacing)
    f.create_dataset('Folder',data=save_dir + file_name)
    f.close()

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
'''
Code below is from OULU lab. It includes how they did the preprocessing and extract knee from 
images
'''
def image_preprocessing_oulu(data_folder,file):
    localizer = KneeLocalizer()

    bbox = worker(file, data_folder, localizer) # output a string
    patch_left, patch_right = read_file_oulu(os.path.join(data_folder, file), bbox)
    return patch_left,patch_right



def read_file_oulu(file_path,bbox,sizemm=140,pad=300):
    '''
    :param file_path: file path the DICOM Data
    :param bbox: file name + box frame corrdinates as a list
    :param sizemm: size
    :param pad: padding size
    :return: pixel data of left knee and right knee
    '''
    data = dicom.read_file(file_path)

    bbox = bbox.split(' ')
    bbox = np.array([int(i) for i in bbox[1:]])
    print(bbox)
    if bbox[0] == -1:
        return None,None
    # process_xray
    # get data from Dicom file
    raw_img = data.pixel_array
    r_,c_ = raw_img.shape
    img = interpolate_resolution(data).copy()
    r,c = img.shape
    ratio_r = r / r_
    ratio_c = c / c_
    img = global_contrast_normalization(img)
    img = hist_truncation(img)
    # define spacing, sizemm, pad
    tmp = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad))
    tmp[pad:pad + img.shape[0], pad:pad + img.shape[1]] = img
    I = tmp
    # left knee coordinates
    x1, y1, x2, y2 = bbox[:4]  # apply padding to the frame of knee

    cx = x1 + (x2 - x1) // 2  # compute center of x
    cy = y1 + (y2 - y1) // 2  # compute cneter of y
    # time the ratio
    cx = int(cx * ratio_c)+ pad
    cy = int(cy * ratio_r) + pad
    x1 = cx - 512
    x2 = cx + 512
    y1 = cy - 512
    y2 = cy + 512
    # compute frame corrdinates

    patch_left = I[y1:y2, x1:x2]
    # right knee coordinates
    x1, y1, x2, y2 = bbox[4:]

    cx = x1 + (x2 - x1) // 2  # compute center of x
    cy = y1 + (y2 - y1) // 2  # compute cneter of y
    # time the ratio
    cx = int(cx * ratio_c) + pad
    cy = int(cy * ratio_r) + pad

    x1 = cx - 512
    x2 = cx + 512
    y1 = cy - 512
    y2 = cy + 512
    # compute frame corrdinates
    patch_right = I[y1:y2, x1:x2]
    print('({},{})-({},{})'.format(x1, y1, x2, y2))
    return patch_left, patch_right
if __name__ == '__main__':
    read_dicome_and_process()
    print('Finished')