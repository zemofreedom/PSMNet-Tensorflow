from __future__ import print_function
import argparse
import os
import random
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from dataloader import KITTIloader2012 as ls
from dataloader import KITTILoader as DA
from dataloader import preprocess
from model import *


#设置输入的参数
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--datatype', default='2012',
                    help='datapath')
parser.add_argument('--datapath', default='../data_stereo_flow/training/', help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

print('Called with args:')
print(args)

if args.datatype == '2015':
   from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
   from dataloader import KITTIloader2012 as ls

#读取数据路径（默认读取2012的数据，返回图片的路径）
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

#读数据，并且讲数据以12个batch进行封装，并且图片全部shuffle
#TrainImgLoader是一个list包括了imgL_crop, imgR_crop, disp_crop_L，list[i]表示一个batch_size

#kitti2012_data = DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True)

TrainImgLoader = DA.ImgLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         BATCH_SIZE = 1)

#TestImgLoader = DA.ImgLoader(
#         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
#         BATCH_SIZE= 8)



"""
#读取已有的模型（后续在写）
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
"""

"""
def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR)

        pred_disp = output3.data.cpu()

        #computing 3-px error#
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)+(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        return 1-(float(torch.sum(correct))/float(len(index[0])))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
       lr = 0.001
    else:
       lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
"""

def main():
        with tf.Session() as sess:
            max_acc=0
            max_epo=0
            start_full_time = time.time()
        #导入模型
            model = Model(sess, height=128, weight=256, batch_size=1, max_disp=args.maxdisp)

            for epoch in range(1, args.epochs+1):
               total_train_loss = 0
               total_test_loss = 0
	       #adjust_learning_rate(optimizer,epoch)
           
                   ## training ##
               for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
                   start_time = time.time() 
                   #print('imgL_crop.shape:',imgL_crop.shape)
                   #print('imgR_crop.shape:',imgR_crop.shape)
                   #print('disp_crop_L.shape:',disp_crop_L.shape)
                   train_loss = model.train(imgL_crop,imgR_crop, disp_crop_L)
                   print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, train_loss, time.time() - start_time))
                   total_train_loss += train_loss
               print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
	   
                   ## Test ##
"""
               for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                   test_loss = test(imgL,imgR, disp_L)
                   print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
                   total_test_loss += test_loss


	       print('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100))
	       if total_test_loss/len(TestImgLoader)*100 > max_acc:
		    max_acc = total_test_loss/len(TestImgLoader)*100
		    max_epo = epoch
	       print('MAX epoch %d total test error = %.3f' %(max_epo, max_acc))

	   #SAVE
	       savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
	       torch.save({
		        'epoch': epoch,
		        'state_dict': model.state_dict(),
		        'train_loss': total_train_loss/len(TrainImgLoader),
		        'test_loss': total_test_loss/len(TestImgLoader)*100,
		    }, savefilename)
	
            print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
	    print(max_epo)
	    print(max_acc)
"""

if __name__ == '__main__':
   main()
