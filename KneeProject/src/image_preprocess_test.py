import numpy as np
import pandas as pd
import os
import time
import shutil
import pydicom as dicom
import sys
sys.path.append('../')
from src.utils import *


summary = pd.read_csv('summary.csv')
print(summary.head())
