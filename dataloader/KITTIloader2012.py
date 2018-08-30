#import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

#function:返回图片的路径

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, shuffle=True):

  left_fold  = 'colored_0/'
  right_fold = 'colored_1/'
  disp_noc   = 'disp_noc/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
  if shuffle:
      random.shuffle(image)
      train = image[:]
      val   = image[160:]
  else:
      image.sort()  
      train = image[:]
      val   = image[160:]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train = [filepath+disp_noc+img for img in train]


  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val = [filepath+disp_noc+img for img in val]

  return left_train, right_train, disp_train, left_val, right_val, disp_val

if __name__ == '__main__':
    left_train, right_train, disp_train, left_val, right_val, disp_val=dataloader('/home/wpj/code/tensorflow/Stereo Match/data_stereo_flow/training/')
    print('~~~~~~~~~~~left_train~~~~~~~~~~~~~~~')
    print(left_train)
    print('~~~~~~~~~~~right_train~~~~~~~~~~~~~~~')
    print(len(right_train))
    print('~~~~~~~~~~~disp_train~~~~~~~~~~~~~~~')
    print(len(disp_train))
    print('~~~~~~~~~~~left_val~~~~~~~~~~~~~~~')
    print(len(left_val))
    print('~~~~~~~~~~~right_val~~~~~~~~~~~~~~~')
    print(len(right_val))
    print('~~~~~~~~~~~disp_val~~~~~~~~~~~~~~~')
    print(len(disp_val))
