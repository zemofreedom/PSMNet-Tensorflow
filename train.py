from __future__ import print_function
import argparse
import os
import tensorflow as tf
import numpy as np
import time
from dataloader.load_SceneFlow import DataLoaderSceneFlow
from models.model import *
import cv2


#设置输入的参数
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=144,
                    help='maxium disparity')
parser.add_argument('--batch', type=int ,default=3,
                    help='batch_size')	
parser.add_argument('--datatype', default='SceneFlow',
                    help='datapath')
parser.add_argument('--datapath', default='../SceneFlow/', help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='',
                    help='load model')
parser.add_argument('--savemodel', default='./ckpt_SF/PSMNet.ckpt',
                    help='save model')

args = parser.parse_args()

print('Called with args:')
print(args)

if args.datatype == '2015':
    left_img = args.datapath + 'image_2/'
    right_img = args.datapath + 'image_3/'
    disp_img = args.datapath + 'disp_occ_0/'   
elif args.datatype == '2012':
    left_img = args.datapath + 'image_0/'
    right_img = args.datapath + 'image_1/'
    disp_img = args.datapath + 'disp_occ/'
elif args.datatype == 'SceneFlow':
    img_path = args.datapath + 'image.mat'
    disp_path = args.datapath + 'disparity.mat'    

#读取数据路径（默认读取2015的数据）
start_loaddata = time.time()
dg = DataLoaderSceneFlow(path="train.lst", batch_size=args.batch, max_disp=args.maxdisp)
print('Load data time = %.2f' %(time.time() - start_loaddata))

if not os.path.exists('./ckpt_SF/'):
    os.mkdir('./ckpt_SF/')

if not os.path.exists('./logs/'):
    os.mkdir('./logs/')
    
if not os.path.exists('./gray/'):
    os.mkdir('./gray/')
if not os.path.exists('./rainbow/'):
    os.mkdir('./rainbow/')


def main():
    with tf.Session() as sess:

        start_load_time = time.time()
        counter = 1
    #导入模型
        model = Model(sess, height=256, weight=512, batch_size=args.batch, max_disp=args.maxdisp)
        saver = tf.train.Saver()
        if args.loadmodel:
            saver.restore(sess, args.loadmodel)

        print('Load model time = %.2f' %(time.time() - start_load_time))
        for epoch in range(1, args.epochs+1):
            total_train_loss = 0
            total_test_loss = 0
            total_test_error = 0
	        #adjust_learning_rate(optimizer,epoch)
            if epoch>30:
                model.lr = 0.0001       
               ## training ##
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=True)):
                start_time = time.time() 
                   #print('imgL_crop.shape:',imgL_crop.shape)
                   #print('imgR_crop.shape:',imgR_crop.shape)
                   #print('disp_crop_L.shape:',disp_crop_L.shape)
                train_loss = model.train(imgL_crop,imgR_crop, disp_crop_L, counter)
                print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss
                counter += 1
            avg_loss = total_train_loss / (7556 // args.batch)
            print('epoch %d avg training loss = %.3f' % (epoch, avg_loss))
            
            saver.save(sess, args.savemodel, global_step=epoch)

            #total_train_loss = 0
            total_pred = []
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=False)):
                start_time = time.time()
                pred, test_loss = model.test(imgL_crop, imgR_crop, disp_crop_L)
                total_pred.append(pred)
                error = np.mean(np.fabs(pred-disp_crop_L))
                print('Iter %d testing loss = %.3f , time = %.2f, error = %.2f' % (batch_idx, test_loss, time.time() - start_time, error))
                total_test_loss += test_loss
                total_test_error += error

            avg_loss = total_test_loss / (1108 // args.batch)
            print('epoch %d avg testing loss = %.3f' % (epoch, avg_loss))
            avg_error = total_test_error / (1108 // args.batch)
            print('epoch %d avg testing loss = %.3f' % (epoch, avg_error))
            
            pred = np.array(total_pred).reshape((-1,256,512))
            
            for i in range(pred.shape[0]):
                pred_disp = (pred[i] * 255 / pred[i].max()).astype(np.uint8)
                #pred_disp = np.squeeze(pred_disp,axis=0)
                path1 = './gray/pred_disp_' + str(i) + '.png'
                cv2.imwrite(path1, pred_disp)
                pred_rainbow = cv2.applyColorMap(pred_disp, cv2.COLORMAP_JET)
                path2 = './rainbow/pred_disp_' + str(i) + '.png'
                cv2.imwrite(path2, pred_rainbow)
        saver.save(sess, args.savemodel)
	   
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
