# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:19:29 2019

@author: Administrator
"""

import re
import numpy as np
import sys
import cv2
 

def readPFM(path):
    file = open(path,'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    #print(header)
    #print(str(file.readline(),encoding='gbk'))
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    #dim_match = re.match(r'^(\d+)\s(\d+)\s$', str(file.readline(),encoding='gbk'))
    dim_match = str(file.readline(),encoding='gbk')
    #print(dim_match.split())
    if dim_match:
        width, height = list(map(int, dim_match.split()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

if __name__ == '__main__':
    f = '0000_left.pfm'
    img = readPFM(f)
    print(img.max()) #H*W
    im = img.astype(np.uint8)
    pred_rainbow = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    cv2.imshow('img_gray',im)
    cv2.imshow('img_rainbow',pred_rainbow)
    cv2.imwrite('pred_gray.png', im)
    cv2.waitKey(0)
    