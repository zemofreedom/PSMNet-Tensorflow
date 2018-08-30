import os
import numpy as np
import random
from PIL import Image, ImageOps
import numpy as np
from dataloader import preprocess


#主要是用来做数据预处理
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(object):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        #print(self.left)
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)


        if self.training:  
           w, h = left_img.size
           th, tw = 128, 256
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
           #对视差图/256可以得到视差值，如果为0，表示那个位置的像素没有groundtruth
           #dataL是一个(256,512)大小矩阵
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]
           
           #processed = preprocess.get_transform(augment=False)  
           left_img   = preprocess.scale_crop(left_img)
           right_img  = preprocess.scale_crop(right_img)

           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-1232, h-368, w, h))
           right_img = right_img.crop((w-1232, h-368, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1232, h-368, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

           #processed = preprocess.get_transform(augment=False)  
           left_img       = preprocess.scale_crop(left_img)
           right_img      = preprocess.scale_crop(right_img)

           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)

#构建mini-batch
def ImgLoader(data=None, BATCH_SIZE=16):
    n=len(data)
    left_img, _, dataL = data[0]
    #print(dataL.shape)
    HEIGHT,WIDTH,CHANNELS = left_img.shape
    mini_batches = []
    imgL = np.zeros([n,HEIGHT,WIDTH,CHANNELS]);
    imgR = np.zeros([n,HEIGHT,WIDTH,CHANNELS]);
    imgD = np.zeros([n,HEIGHT,WIDTH]);

    for i in range(len(data)):
        imgL[i],imgR[i],imgD[i] = data[i]

    batch_num = int(n / BATCH_SIZE)
    for k in range(0, batch_num):
        mini_batch_L = imgL[k*BATCH_SIZE:(k+1)*BATCH_SIZE]
        mini_batch_R = imgR[k*BATCH_SIZE:(k+1)*BATCH_SIZE]
        mini_batch_D = imgD[k*BATCH_SIZE:(k+1)*BATCH_SIZE]
        mini_batches.append((mini_batch_L, mini_batch_R, mini_batch_D))
    
    if n % BATCH_SIZE != 0:
        mini_batch_L = imgL[batch_num*BATCH_SIZE:]
        mini_batch_R = imgR[batch_num*BATCH_SIZE:]
        mini_batch_D = imgD[batch_num*BATCH_SIZE:]
        mini_batches.append((mini_batch_L, mini_batch_R, mini_batch_D))
#    print(np.array(mini_batches))
    
    return mini_batches

if __name__ == '__main__':
    import KITTIloader2012 as ls
    path = '/home/wpj/code/tensorflow/Stereo Match/data_stereo_flow/training/'
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(path)
    kitti2012_data = myImageFloder(all_left_img,all_right_img,all_left_disp, True)
    print(len(ImgLoader(kitti2012_data)))
