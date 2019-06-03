import random
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import numbers
import numpy as np


class CenterCrop(object):
    """
    Performs center crop of an image of a certain size.
    Modified version from torchvision.

    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):

        w, h = img.shape[0],img.shape[1]
        tw, th, = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return img[y1:y1+th,x1:x1+tw] #img.crop((x1, y1, x1 + tw, y1 + th))

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        # Note: image_data_format is 'channel_last'
        # assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = self.size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx)]
'''
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    #assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

'''
