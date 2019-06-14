# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:59:01 2019

@author: Administrator
"""

import numpy as np
import random
import cv2
import os

import h5py


class DataLoaderSceneFlow(object):
    def __init__(self, img_path, disp_path, batch_size, patch_size=(256, 512), max_disp=129):
        self.img_path = img_path
        self.disp_path = disp_path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_disp = max_disp
        img_m = h5py.File(self.img_path)
        disp_m = h5py.File(self.disp_path)

        left_img = img_m["left_img"][:]
        right_img = img_m["right_img"][:]
        disp_img = disp_m["disp_img"][:]
        
        left_img=left_img.transpose() #(N,H,W,3) array number=8864
        right_img=right_img.transpose()#(N,H,W,3) array
        disp_img=disp_img.transpose()#(N,H,W) array
        print("load data success!!!")

        self.num, self.heigh, self.weight = disp_img.shape
        
# =============================================================================
#         state = np.random.get_state()
#         np.random.shuffle(left_img)     
#         np.random.set_state(state)
#         np.random.shuffle(right_img)
#         np.random.set_state(state)
#         np.random.shuffle(disp_img)
# =============================================================================

        self.val_left = left_img[:1108]
        self.val_right = right_img[:1108]
        self.val_labels = disp_img[:1108]

        
        self.shuffled_left_data = left_img[1108:]
        self.shuffled_right_data = right_img[1108:]
        self.shuffled_labels = disp_img[1108:]

    def generator(self, is_training=True):

        
        

        if is_training:
            state = np.random.get_state()
            np.random.shuffle(self.shuffled_left_data)     
            np.random.set_state(state)
            np.random.shuffle(self.shuffled_right_data)
            np.random.set_state(state)
            np.random.shuffle(self.shuffled_labels)
        


      
        print("start making data!!!")
        
        if is_training:
            for j in range((self.num-1108) // self.batch_size):
                left, right, label = self.load_batch(self.shuffled_left_data[j * self.batch_size: (j + 1) * self.batch_size],
                                                     self.shuffled_right_data[
                                                     j * self.batch_size: (j + 1) * self.batch_size],
                                                     self.shuffled_labels[j * self.batch_size: (j + 1) * self.batch_size],
                                                     is_training)
                left = np.array(left)
                right = np.array(right)
                label = np.array(label)
                yield left, right, label
        else:
            for j in range(1108 // self.batch_size):
                left, right, label = self.load_batch(self.val_left[j * self.batch_size: (j + 1) * self.batch_size],
                                                     self.val_right[j * self.batch_size: (j + 1) * self.batch_size],
                                                     self.val_labels[j * self.batch_size: (j + 1) * self.batch_size],
                                                     is_training)
                left = np.array(left)
                right = np.array(right)
                label = np.array(label)
                yield left, right, label

    def load_batch(self, left, right, labels, is_training):
        batch_left = []
        batch_right = []
        batch_label = []
        for i in range(left.shape[0]):
            if is_training:
                crop_x = random.randint(0, self.heigh - self.patch_size[0] -1)
                crop_y = random.randint(0, self.weight - self.patch_size[1] -1)
            else:
                crop_x = random.randint(0, self.heigh - self.patch_size[0] -1)
                crop_y = random.randint(0, self.weight - self.patch_size[1] -1)

            x = left[i, crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1], :]
            x = self.mean_std(x)
            batch_left.append(x)


            y = right[i, crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1], :]
            y = self.mean_std(y)
            batch_right.append(y)


            z = labels[i, crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1]]
            z[z > (self.max_disp-1)] = self.max_disp - 1
            batch_label.append(z)
        return batch_left, batch_right, batch_label

    @staticmethod
    def mean_std(inputs):
        inputs = np.float32(inputs) / 255.
        inputs[:, :, 0] -= 0.485
        inputs[:, :, 0] /= 0.229
        inputs[:, :, 1] -= 0.456
        inputs[:, :, 1] /= 0.224
        inputs[:, :, 2] -= 0.406
        inputs[:, :, 2] /= 0.225
        return inputs
