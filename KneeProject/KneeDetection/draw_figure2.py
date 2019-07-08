'''

Generate model predict bbox
'''
import pandas as pd
import numpy as np
import os
import time
import h5py
import pydicom as dicom
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sys
Month = str(sys.argv[1])
df = pd.read_csv('output{}.csv'.format(Month))

def preprocessing(data_path,reshape):
    dicom_img = dicom.dcmread(data_path)
    img = dicom_img.pixel_array.astype(float)
    img = (np.maximum(img, 0) / img.max()) * 255.0
    row, col = img.shape
    img = cv2.resize(img, (reshape, reshape), interpolation=cv2.INTER_CUBIC)
    ratio_x = reshape / col
    ratio_y = reshape / row
    return img

def drawFigure(img,labels,preds,f_name,folder):
    '''
    draw a png figure with rect of ground truth and prediction
    col == x, row == y
    :param img:
    :param labels:
    :param preds:
    :param f_name:
    :return:
    '''
    fig, ax = plt.subplots(1)
    row, col = img.shape
    ax.imshow(img)


    # draw true patch
    if labels is not None:
        labels = labels * row
        x1, y1, x2, y2 = labels[:4]
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        x1, y1, x2, y2 = labels[4:]
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect2)
    if preds is not None:
        # draw predict patch
        preds = preds * row
        x1, y1, x2, y2 = preds[:4]
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect1)
        x1, y1, x2, y2 = preds[4:]
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect2)
    # save image
    plt.savefig(os.path.join(folder, f_name), dpi=300)
    plt.close()
output_folder = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/test/bbox_pred{}/'.format(Month)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for idx, row in df.iterrows():
    row = row.tolist()
    bbox = np.array(row[:8])
    img = preprocessing(row[-1],898)
    f_name = row[-1]
    f_name = f_name.split('/')[-6:-1]
    f_name = '_'.join(f_name) + '.png'
    print(f_name)
    drawFigure(img,None,bbox,f_name,output_folder)