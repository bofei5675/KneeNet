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
def image_preprocessing(file_path = '../data/9000296'):
    '''

    :param file_path:
    :return:
    '''
    # read data from DICOM file
    data = dicom.read_file(file_path)
    img = interpolate_resolution(data) # get fixed resolution
    img_before = img.copy()
    rows, cols = img.shape
    # get center part of image if image is large enough
    if rows >= 2048 and cols >= 2048:
        img = get_center_image(img)
    else:
        img,_,_ = padding(img)
        img = get_center_image(img) # after padding get the center of image
    # apply normalization
    img = global_contrast_normalization(img)
    # apply hist truncation
    img = hist_truncation(img)

    return img,data,img_before

def interpolate_resolution(image_dicom, scaling_factor=0.2):
    '''
    Obtain fixed resolution from image dicom
    :param image_dicom:
    :param scaling_factor:
    :return:
    '''
    image_array = image_dicom.pixel_array

    x = image_dicom[0x28, 0x30].value[0]
    y = image_dicom[0x28, 0x30].value[1]

    image_array = ndimage.zoom(image_array, [x / scaling_factor, y / scaling_factor])
    return image_array
def get_center_image(img,img_size = (2048,2048)):
    '''
    Get the center of image
    :param img:
    :param img_size:
    :return:
    '''
    rows,cols = img.shape
    center_x = rows // 2
    center_y = cols // 2
    img_crop = img[center_x - img_size[0] // 2: center_x + img_size[0] // 2,
                   center_y - img_size[1] // 2: center_y + img_size[1] // 2]
    return img_crop

def padding(img,img_size = (2048,2048)):
    '''
    Padding image array to a specific size
    :param img:
    :param img_size:
    :return:
    '''
    rows,cols = img.shape
    x_padding = img_size[0] - rows
    y_padding = img_size[1] - cols
    if x_padding > 0:
        before_x,after_x = x_padding // 2, x_padding - x_padding // 2
    else:
        before_x,after_x = 0,0
    if y_padding > 0:
        before_y,after_y = y_padding // 2, y_padding - y_padding // 2
    else:
        before_y,after_y = 0,0
    return np.pad(img,((before_x,after_x),(before_y,after_y)),'constant'),before_x,before_y


def global_contrast_normalization(img, s=1, lambda_=10, epsilon=1e-8):
    '''
    Apply global contrast normalization based on image array.
    :param img:
    :param s:
    :param lambda_:
    :param epsilon:
    :return:
    '''
    # replacement for the loop
    X_average = np.mean(img)
    print('Mean: ', X_average)
    img_center = img - X_average

    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lambda_ + np.mean(img_center ** 2))

    img = s * img_center / max(contrast, epsilon)

    # scipy can handle it
    return img
def hist_truncation(img,cut_min=5,cut_max = 95):
    '''
    Apply 5th and 99th truncation on the figure.
    :param img:
    :param cut_min:
    :param cut_max:
    :return:
    '''
    lim1,lim2 = np.percentile(img,[cut_min, cut_max])
    img_ = img.copy()
    img_[img < lim1] = lim1
    img_[img > lim2] = lim2
    return img_


def extract_knee(image_array, side):
    '''
    Extrack knee part from image array
    :param image_array:
    :param side: 0: left knee; 1: right knee
    :return:
    '''
    print('Dimensions of image: ', image_array.shape)

    # Compute the sum of each row and column
    col_sums = np.sum(image_array, axis=0)
    row_sums = np.sum(image_array, axis=1)

    # Row index for cropping is centered at the minimum of the row_sums array
    row_start = np.argmin(row_sums) - 512
    row_end = np.argmin(row_sums) + 512
    print('Row Indices Original: ', row_start, row_end)

    # However, if either the start or end of the row values is beyond the original image array shape
    # We center the cropped image at the center row of the original image array
    if row_start < 0 or row_end > (image_array.shape[0] - 1):
        row_start = round(image_array.shape[0] / 2) - 512
        row_end = round(image_array.shape[0] / 2) + 512

    print('Row Indices Final: ', row_start, row_end)

    # For right knee, crop columns to be centered at the maximum sum of the LHS of original image array
    # Shift over by 500 columns in edge cases with white outer bars
    if side == 1:
        col_center = 500 + np.argmax(col_sums[500:round(col_sums.shape[0] / 2)])
        print('Column Indices for Right Original: ', col_center - 512, col_center + 512)

        # If column is below original image array size, then start cropping on left hand border and go out 1024 columns
        if (col_center - 512) < 0:
            print('Column Indices for Right Final: ', 0, 1024)
            image_array = image_array[row_start:row_end, :1024]

        else:
            image_array = image_array[row_start:row_end, (col_center - 512):(col_center + 512)]
            print('Column Indices for Right Final: ', col_center - 512, col_center + 512)

    # For left knee, crop columns to be centered at the maximum sum of the RHS of original image array
    # Shift over by 500 columns in edge cases with white outer bars
    if side == 0:
        col_center = round(col_sums.shape[0] / 2) + np.argmax(
            col_sums[round(col_sums.shape[0] / 2):col_sums.shape[0] - 500])
        print('Column Indices for Left Original: ', col_center - 512, col_center + 512)

        # If column is above original image array size, then start cropping on right hand border and go in 1024 columns
        if (col_center + 512) > (image_array.shape[1] - 1):
            print('Column Indices for Left Final: ', image_array.shape[1] - 1024, image_array.shape[1] - 1)
            image_array = image_array[row_start:row_end, image_array.shape[1] - 1024:]

        else:
            image_array = image_array[row_start:row_end, (col_center - 512):(col_center + 512)]
            print('Column Indices for Left Final: ', col_center - 512, col_center + 512)
    return image_array

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
def read_dicome_and_convert(content_file_path='/gpfs/data/denizlab/Datasets/OAI_original/',month = '00m',save_dir = '/gpfs/data/denizlab/Users/bz1030/test/OAI_original/'):
    '''
    read all dicome and convert them to fix resolution
    :param content_file_path:
    :param month:
    :return:
    '''
    content_file_path = os.path.join(content_file_path, month)
    file_name = 'contents.csv'
    count = 0
    save_dir  = os.path.join(save_dir,month)
    print('Save to ',save_dir)
    with open(os.path.join(content_file_path, file_name), 'r') as f:
        next(f)  # skip first row
        for line in f:
            line = line.rstrip().replace('"', '').split(',')  # split each line by csv
            data_path, patientID, studyDate, barCode, description = line[0], line[1], line[2], line[3], line[4]
            description = description.rstrip().replace('"', '').replace(' ', '').split(
                '^')  # split fields inside description
            # only look at XRAY and Knee data
            if description[1] == 'XRAY' and description[-1] == 'KNEE':
                data_path = content_file_path + '/' + data_path.replace('"', '')
                data_files = os.listdir(data_path)
                for data_file in data_files:
                    img, data, img_before = image_preprocessing(os.path.join(data_path, data_file))
                    data.PixelData = img.tobytes()
                    file_path = os.path.join(save_dir,patientID,studyDate,barCode)
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    data.save_as(os.path.join(file_path,data_file))

                    count += 1

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
                    img,data,img_before = image_preprocessing(os.path.join(data_path,data_file))
                    left_svm, right_svm = image_preprocessing_oulu(data_path,data_file)
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
        print('More figures')
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
    if bbox[0] == -1:
        return None,None
    print(bbox)
    # process_xray
    # get data from Dicom file
    img = np.frombuffer(data.PixelData, dtype=np.uint16).copy().astype(np.float64)
    img = img.reshape((data.Rows, data.Columns))
    x1, y1, x2, y2 = bbox[:4]
    patch_left = img[y1:y2,x1:x2]
    x1, y1, x2, y2 = bbox[4:]
    patch_right = img[y1:y2,x1:x2]

    return patch_left,patch_right

    cut_min = 5
    cut_max = 99
    multiplier = 65535

    img = img.copy()
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2

    img -= lim1
    img /= img.max()
    img *= multiplier

    img = img.astype(np.float)
    # define spacing, sizemm, pad
    try:
        spacing = float(data.ImagerPixelSpacing[0])
    except AttributeError:
        spacing = 0.2
    sizepx = int(np.round(sizemm / spacing))

    tmp = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad))
    tmp[pad:pad + img.shape[0], pad:pad + img.shape[1]] = img
    I = tmp
    # left knee coordinates
    x1, y1, x2, y2 = bbox[:4] + pad  # apply padding to the frame of knee

    cx = x1 + (x2 - x1) // 2  # compute center of x
    cy = y1 + (y2 - y1) // 2  # compute cneter of y

    x1 = cx - 512
    x2 = cx + 512
    y1 = cy - 512
    y2 = cy + 512
    # compute frame corrdinates

    patch = I[y1:y2, x1:x2]
    patch_left = patch.astype(np.uint16)

    # right knee coordinates
    x1, y1, x2, y2 = bbox[4:] + pad

    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2

    x1 = cx - 512
    x2 = cx + 512
    y1 = cy - 512
    y2 = cy + 512
    # compute frame corrdinates
    patch = I[y1:y2, x1:x2]
    print('({},{})-({},{})'.format(x1, y1, x2, y2))
    patch_right = patch.astype(np.uint16)

    return patch_left, patch_right

if __name__ == '__main__':
    read_dicome_and_convert()
    print('Finished')