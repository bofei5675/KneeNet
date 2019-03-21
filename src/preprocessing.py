import argparse
import pandas as pd
import os
from pathlib import Path
import pydicom as dicom
import numpy
import h5py
import numpy as np
import cv2
from tqdm import tqdm

from oulukneeloc import SVM_MODEL_PATH
from oulukneeloc.proposals import (read_dicom, get_joint_y_proposals,
                                   preprocess_xray)
import matplotlib.pyplot as plt
from detector import KneeLocalizer,worker






if __name__ == '__main__':

    print('Finished')