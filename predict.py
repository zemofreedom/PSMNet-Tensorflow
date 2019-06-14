from __future__ import print_function
import argparse
import os
import tensorflow as tf
import numpy as np
import time
from dataloader.data_loader import DataLoaderKITTI
from models.model import *
import cv2


#set para
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--batch', type=int ,default=1,
                    help='batch_size')	
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='../KITTI_2015/training/', help='datapath')
parser.add_argument('--loadmodel', default='./ckpt_KITTI/PSMNet.ckpt-30',
                    help='load model')
parser.add_argument('--leftimg', default='image_2/000000_10.png',
                    help='left image')
parser.add_argument('--rightimg', default='image_3/000000_10.png',
                    help='right image')

args = parser.parse_args()

print('Called with args:')
print(args)

def main():

    height = 368 #544 #368
    weight = 1232 #960 #1232
    left_img = args.datapath+args.leftimg
    right_img = args.datapath+args.leftimg


    with tf.Session() as sess:


        img_L = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)
        img_L = cv2.resize(img_L, (weight, height))
        img_R = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)
        img_R = cv2.resize(img_R, (weight, height))		

        img_L = DataLoaderKITTI.mean_std(img_L)
        img_L = np.expand_dims(img_L, axis=0)
        img_R = DataLoaderKITTI.mean_std(img_R)
        img_R = np.expand_dims(img_R, axis=0)
		
        PSMNet = Model(sess, height=height, weight=weight, batch_size=args.batch, max_disp=args.maxdisp)
        saver = tf.train.Saver()
        saver.restore(sess, args.loadmodel)
		
        pred = PSMNet.predict(img_L, img_R)
        pred = np.squeeze(pred,axis=0)
        print(pred.shape)
        print(pred.max())
        #np.save('pred.npy',pred)
        
        pred_disp = pred.astype(np.uint8)
        print(pred_disp.shape)
        #pred_disp = np.squeeze(pred_disp,axis=0)
        cv2.imwrite('pred_disp.png', pred_disp)
        pred_rainbow = cv2.applyColorMap(pred_disp, cv2.COLORMAP_RAINBOW)
        cv2.imwrite('pred_rainbow.png', pred_rainbow)


if __name__ == '__main__':
    main()
