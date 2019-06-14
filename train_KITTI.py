from __future__ import print_function
import argparse
import os
import tensorflow as tf
import numpy as np
import time
from dataloader.data_loader import DataLoaderKITTI
from models.model import *
import cv2


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--batch', type=int ,default=3,
                    help='batch_size')	
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='../KITTI_2015/training/', help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='',
                    help='load model')
parser.add_argument('--savemodel', default='./ckpt_KITTI/PSMNet.ckpt',
                    help='save model')

args = parser.parse_args()

print('Called with args:')
print(args)

if args.datatype == '2015':
    left_img = args.datapath + 'image_2/'
    right_img = args.datapath + 'image_3/'
    disp_img = args.datapath + 'disp_occ_0/'   
elif args.datatype == '2012':
    left_img = args.datapath + 'colored_0/'
    right_img = args.datapath + 'colored_1/'
    disp_img = args.datapath + 'disp_occ/'   


h=256
w=512

dg = DataLoaderKITTI(left_img, right_img, disp_img, args.batch, patch_size=[h, w])

if not os.path.exists('./ckpt_KITTI/'):
    os.mkdir('./ckpt_KITTI/')


if not os.path.exists('./gray_KITTI/'):
    os.mkdir('./gray_KITTI/')
if not os.path.exists('./rainbow_KITTI/'):
    os.mkdir('./rainbow_KITTI/')

if not os.path.exists('./logs/'):
    os.mkdir('./logs/')

"""
#if args.loadmodel is not None:
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

        start_full_time = time.time()
        counter = 1

        model = Model(sess, height=h, weight=w, batch_size=args.batch, max_disp=args.maxdisp)
        saver = tf.train.Saver()
        model.lr = 1e-4
        if args.loadmodel:
            saver.restore(sess, args.loadmodel)
        for epoch in range(1, args.epochs+1):
            if epoch>600:
                model.lr = 1e-5
            total_train_loss = 0
            total_test_loss = 0
            total_test_error = 0
            total_test_val_err = 0
	        #adjust_learning_rate(optimizer,epoch)
       
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
            avg_loss = total_train_loss / (200 // args.batch)
            print('epoch %d avg training loss = %.3f' % (epoch, avg_loss))
            if epoch % 10 == 0:
                saver.save(sess, args.savemodel, global_step=epoch)

            #total_train_loss = 0
            total_pred = []
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=False)):
                start_time = time.time()
                pred, test_loss = model.test(imgL_crop, imgR_crop, disp_crop_L)
                total_pred.append(pred)
                epe = np.mean(np.fabs(pred-disp_crop_L))
                val_err = np.sum(np.fabs(pred-disp_crop_L)>3)/(3*h*w)
                print('Iter %d testing loss = %.3f , time = %.2f, epe_error = %.2f, val_error = %.4f' % (batch_idx, test_loss, time.time() - start_time, epe, val_err))
                #print('Iter %d testing loss = %.3f , time = %.2f, error = %.2f' % (batch_idx, test_loss, time.time() - start_time, error))
                total_test_loss += test_loss
                total_test_error += epe
                total_test_val_err += val_err

            avg_loss = total_test_loss / (40 // args.batch)
            print('epoch %d avg testing loss = %.3f' % (epoch, avg_loss))
            avg_error = total_test_error / (40 // args.batch)
            print('epoch %d avg testing mean error = %.3f' % (epoch, avg_error))
            avg_val_err = total_test_val_err / (40 // args.batch)
            print('epoch %d avg testing 3 pixel error = %.3f' % (epoch, avg_val_err))
                    
            pred = np.array(total_pred).reshape((-1,h,w))
        
            for i in range(pred.shape[0]):
                pred_disp = pred[i].astype(np.uint8)
                #pred_disp = np.squeeze(pred_disp,axis=0)
                path1 = './gray_KITTI/pred_disp_' + str(i) + '.png'
                cv2.imwrite(path1, pred_disp)
                pred_rainbow = cv2.applyColorMap(pred_disp, cv2.COLORMAP_JET)
                path2 = './rainbow_KITTI/pred_disp_' + str(i) + '.png'
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
