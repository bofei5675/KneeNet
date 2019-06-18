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
    photoInterpretation = data[0x28,0x04].value # return a string of photometric interpretation
    #print('######### PHOTO INTER {} #########'.format(photoInterpretation))
    if photoInterpretation not in ['MONOCHROME2','MONOCHROME1']:
        raise ValueError('Wrong Value of Photo Interpretation: {}'.format(photoInterpretation))
    img = interpolate_resolution(data).astype(np.float64) # get fixed resolution
    img_before = img.copy()
    if photoInterpretation == 'MONOCHROME1':
        img = invert_Monochrome1(img)
    # apply normalization, move into hist_truncation.
    # img = global_contrast_normalization(img)
    # apply hist truncation
    img = hist_truncation(img)
    rows, cols = img.shape
    # get center part of image if image is large enough
    if rows >= 2048 and cols >= 2048:
        img = get_center_image(img)
    else:
        img,_,_ = padding(img)
        img = get_center_image(img) # after padding get the center of image


    return img,data,img_before
def invert_Monochrome1(image_array):
    '''
    Image with dicome attribute [0028,0004] == MONOCHROME1 needs to
    be inverted. Otherwise, our way to detect the knee will not work.

    :param image_array:
    :return:
    '''
    print('Invert Monochrome ')
    print(image_array.shape, np.mean(image_array), np.min(image_array), np.max(image_array))
    # image_array = -image_array + 255.0 # our method
    image_array = image_array.max() - image_array
    print(image_array.shape, np.mean(image_array), np.min(image_array), np.max(image_array))
    return image_array

def interpolate_resolution(image_dicom, scaling_factor=0.2):
    '''
    Obtain fixed resolution from image dicom
    :param image_dicom:
    :param scaling_factor:
    :return:
    '''
    print('Obtain Fix Resolution:')
    image_array = image_dicom.pixel_array
    print(image_array.shape,np.mean(image_array),np.min(image_array),np.max(image_array))
    x = image_dicom[0x28, 0x30].value[0]
    y = image_dicom[0x28, 0x30].value[1]

    image_array = ndimage.zoom(image_array, [x / scaling_factor, y / scaling_factor])
    print(image_array.shape,np.mean(image_array),np.min(image_array),np.max(image_array))
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

def global_contrast_normalization_oulu(img,lim1,multiplier = 255):
    '''
    This part is taken from oulu's lab. This how they did global contrast normalization.
    :param img:
    :param lim1:
    :param multiplier:
    :return:
    '''
    img -= lim1
    img /= img.max()
    img *= multiplier
    return img
def global_contrast_normalization(img, s=1, lambda_=10, epsilon=1e-8):
    '''
    Apply global contrast normalization based on image array.
    Deprecated since it is not working ...
    :param img:
    :param s:
    :param lambda_:
    :param epsilon:
    :return:
    '''
    # replacement for the loop
    print('Global contrast normalization:')
    print(img.shape, np.mean(img), np.min(img), np.max(img))
    X_average = np.mean(img)
    #print('Mean: ', X_average)
    img_center = img - X_average

    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lambda_ + np.mean(img_center ** 2))

    img = s * img_center / max(contrast, epsilon)
    print(img.shape, np.mean(img), np.min(img), np.max(img))
    # scipy can handle it
    return img
def hist_truncation(img,cut_min=5,cut_max = 99):
    '''
    Apply 5th and 99th truncation on the figure.
    :param img:
    :param cut_min:
    :param cut_max:
    :return:
    '''
    print('Trim histogram')
    print(img.shape, np.mean(img), np.min(img), np.max(img))
    lim1,lim2 = np.percentile(img,[cut_min, cut_max])
    img_ = img.copy()
    img_[img < lim1] = lim1
    img_[img > lim2] = lim2
    print(img_.shape, np.mean(img_), np.min(img_), np.max(img_))
    img_ = global_contrast_normalization_oulu(img_,lim1,multiplier=255)
    print(img_.shape, np.mean(img_), np.min(img_), np.max(img_))
    return img_


def extract_knee(image_array, side, offset = None):
    '''
    Extrack knee part from image array
    :param image_array:
    :param side: 0: left knee; 1: right knee
    :param offset: if does not work, you can manually change the shape
    :return:
    '''
    #print('Dimensions of image: ', image_array.shape)

    # Compute the sum of each row and column
    col_sums = np.sum(image_array, axis=0)
    row_sums = np.sum(image_array, axis=1)

    # Row index for cropping is centered at the minimum of the row_sums array
    row_start = np.argmin(row_sums) - 512
    row_end = np.argmin(row_sums) + 512
    #print('Row Indices Original: ', row_start, row_end)

    # However, if either the start or end of the row values is beyond the original image array shape
    # We center the cropped image at the center row of the original image array
    if row_start < 0 or row_end > (image_array.shape[0] - 1):
        row_start = round(image_array.shape[0] / 2) - 512
        row_end = round(image_array.shape[0] / 2) + 512

    #print('Row Indices Final: ', row_start, row_end)

    # For right knee, crop columns to be centered at the maximum sum of the LHS of original image array
    # Shift over by 500 columns in edge cases with white outer bars
    if side == 1:
        col_center = 500 + np.argmax(col_sums[500:round(col_sums.shape[0] / 2)])
        #print('Column Indices for Right Original: ', col_center - 512, col_center + 512)

        # If column is below original image array size, then start cropping on left hand border and go out 1024 columns
        if (col_center - 512) < 0:
            #print('Column Indices for Right Final: ', 0, 1024)
            if offset:
                image_array = image_array[row_start + offset[0]:row_end + offset[0], :1024]
            else:
                image_array = image_array[row_start:row_end, :1024]

        else:
            if offset:
                image_array = image_array[row_start + offset[0]:row_end + offset[0], (col_center - 512) + offset[1]:(col_center + 512)+ offset[1]]
            else:
                image_array = image_array[row_start:row_end, (col_center - 512):(col_center + 512)]

            #print('Column Indices for Right Final: ', col_center - 512, col_center + 512)

    # For left knee, crop columns to be centered at the maximum sum of the RHS of original image array
    # Shift over by 500 columns in edge cases with white outer bars
    if side == 0:
        col_center = round(col_sums.shape[0] / 2) + np.argmax(
            col_sums[round(col_sums.shape[0] / 2):col_sums.shape[0] - 500])
        #print('Column Indices for Left Original: ', col_center - 512, col_center + 512)

        # If column is above original image array size, then start cropping on right hand border and go in 1024 columns
        if (col_center + 512) > (image_array.shape[1] - 1):
            #print('Column Indices for Left Final: ', image_array.shape[1] - 1024, image_array.shape[1] - 1)
            if offset:
                image_array = image_array[row_start + offset[0]:row_end + offset[0], image_array.shape[1] - 1024:]
            else:
                image_array = image_array[row_start :row_end, image_array.shape[1] - 1024:]


        else:
            if offset:
                image_array = image_array[row_start + offset[0]:row_end + offset[0], (col_center - 512) + offset[1]:(col_center + 512) + offset[1]]
            else:
                image_array = image_array[row_start:row_end, (col_center - 512):(col_center + 512)]

            #print('Column Indices for Left Final: ', col_center - 512, col_center + 512)
    return image_array

'''
Code below is from OULU lab. It includes how they did the preprocessing and extract knee from 
images
'''
def process_file(data,pad):
    raw_img = data.pixel_array
    r_, c_ = raw_img.shape
    img = interpolate_resolution(data).astype(np.float64)
    photoInterpretation = data[0x28, 0x04].value  # return a string of photometric interpretation
    # print('######### PHOTO INTER {} #########'.format(photoInterpretation))
    if photoInterpretation not in ['MONOCHROME2', 'MONOCHROME1']:
        raise ValueError('Wrong Value of Photo Interpretation: {}'.format(photoInterpretation))
    elif photoInterpretation == 'MONOCHROME1':
        img = invert_Monochrome1(img)
    r, c = img.shape
    ratio_r = r / r_
    ratio_c = c / c_
    img = hist_truncation(img)
    #img = global_contrast_normalization(img)
    # define spacing, sizemm, pad
    tmp = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad))
    tmp[pad:pad + img.shape[0], pad:pad + img.shape[1]] = img
    return tmp,ratio_c,ratio_r

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
    if -1 in bbox: # if the algorithm says there is no knee in the figure.
        return None,None
    # process_xray
    # get data from Dicom file
    tmp,ratio_c,ratio_r = process_file(data,pad)

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