import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.keras as keras
from .utils import *


class Model:

    def __init__(self, sess, height=256, weight=512, batch_size=16,max_disp = 192):
        self.reg = 1e-4  # TODO
        self.max_disp = max_disp  # TODO
        self.image_size_tf = None
        self.height = height
        self.weight = weight
        self.batch_size = batch_size
        self.sess = sess
        self.lr = 0.001
        self.build_model()



    def build_model(self):
        self.left = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        self.right = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.image_size_tf = tf.shape(self.left)[1:3]

        conv4_left = self.CNN(self.left)
        fusion_left = self.SPP(conv4_left)

        conv4_right = self.CNN(self.right, True)
        fusion_right = self.SPP(conv4_right, True)

        cost_vol = self.cost_vol(fusion_left, fusion_right, self.max_disp)
        outputs = self.CNN3D(cost_vol)
        disps = self.output(outputs)#size of (B, H, W),3out
        self.disps = disps[2]
        #print(self.disps.shape)
        #self.loss = self._smooth_l1_loss(disps[2], self.label)
        self.loss = 0.5*self._smooth_l1_loss(disps[0], self.label)+0.7*self._smooth_l1_loss(disps[1], self.label)+self._smooth_l1_loss(disps[2], self.label)
        

        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        
       
        tf.summary.scalar("loss", self.loss)
        self.merge = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('logs/', self.sess.graph)
        self.train_op = optimizer.minimize(self.loss)
        

        try:
          self.sess.run(tf.global_variables_initializer())
        except:
          self.sess.run(tf.initialize_all_variables())
        


    def _smooth_l1_loss(self, disps_pred, disps_targets, sigma=1.0):
        sigma_2 = sigma ** 2
        disps_diff = disps_pred - disps_targets
        in_disps_diff = disps_diff
        abs_in_disps_diff = tf.abs(in_disps_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_disps_diff, 1. / sigma_2)))
        loss_disps = tf.pow(in_disps_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_disps_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        loss = tf.reduce_mean(tf.reduce_mean(loss_disps,[1,2]))
        return loss

    def CNN(self, bottom, reuse=False):
        with tf.variable_scope('CNN'):
            with tf.variable_scope('conv0'):
                bottom = conv_block(tf.layers.conv2d, bottom, 32, 3, strides=2, name='conv0_1', reuse=reuse, reg=self.reg)
                for i in range(1, 3):
                    bottom = conv_block(tf.layers.conv2d, bottom, 32, 3, name='conv0_%d' % (i+1), reuse=reuse, reg=self.reg)
            with tf.variable_scope('conv1'):
                for i in range(3):
                    bottom = res_block(tf.layers.conv2d, bottom, 32, 3, name='conv1_%d' % (i+1), reuse=reuse, reg=self.reg)
            with tf.variable_scope('conv2'):
                bottom = res_block(tf.layers.conv2d, bottom, 64, 3, strides=2, name='conv2_1', reuse=reuse, reg=self.reg,
                                   projection=True)
                for i in range(1, 16):
                    bottom = res_block(tf.layers.conv2d, bottom, 64, 3, name='conv2_%d' % (i+1), reuse=reuse, reg=self.reg)
            with tf.variable_scope('conv3'):
                bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=2, name='conv3_1', reuse=reuse,
                                   reg=self.reg, projection=True)
                for i in range(1, 3):
                    bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=2, name='conv3_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
            with tf.variable_scope('conv4'):
                for i in range(3):
                    bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=4, name='conv4_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
        return bottom

    def SPP(self, bottom, reuse=False):
        with tf.variable_scope('SPP'):
            branches = []
            for i, p in enumerate([64, 32, 16, 8]):
                branches.append(SPP_branch(tf.layers.conv2d, bottom, p, 32, 3, name='branch_%d' % (i+1), reuse=reuse,
                                           reg=self.reg))
            #if not reuse:                
            conv2_16 = tf.get_default_graph().get_tensor_by_name('CNN/conv2/conv2_16/add:0')
            conv4_3 = tf.get_default_graph().get_tensor_by_name('CNN/conv4/conv4_3/add:0')
            #else:
            #    conv2_16 = tf.get_default_graph().get_tensor_by_name('CNN_1/conv2/conv2_16/add:0')
            #    conv4_3 = tf.get_default_graph().get_tensor_by_name('CNN_1/conv4/conv4_3/add:0')            
            concat = tf.concat([conv2_16, conv4_3] + branches, axis=-1, name='concat')
            with tf.variable_scope('fusion'):
                bottom = conv_block(tf.layers.conv2d, concat, 128, 3, name='conv1', reuse=reuse, reg=self.reg)
                fusion = conv_block(tf.layers.conv2d, bottom, 32, 1, name='conv2', reuse=reuse, reg=self.reg)
        return fusion

    def cost_vol(self, left, right, max_disp=192):
        with tf.variable_scope('cost_vol'):
            disparity_costs = []
            #shape = tf.shape(right) #(N,H,W,F)
            for i in range(max_disp // 4):
                if i > 0:
                    left_tensor = keras.backend.spatial_2d_padding(left[:, :, i:, :], padding=((0, 0), (i, 0)))
                    right_tensor = keras.backend.spatial_2d_padding(right[:, :, :-i, :], padding=((0, 0), (i, 0)))
                    cost = tf.concat([left_tensor, right_tensor], axis=3)
                else:
                    cost = tf.concat([left, right], axis=3)
                disparity_costs.append(cost)
            
            cost_vol = tf.stack(disparity_costs, axis=1)
        return cost_vol

    def CNN3D(self, bottom):
        with tf.variable_scope('CNN3D'):
            for i in range(2):
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_%d' % (i+1), reg=self.reg)

            _3Dconv1 = res_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv1', reg=self.reg)

            _3Dstack = [hourglass('3d', _3Dconv1, [64, 64, 64, 32], [3, 3, 3, 3], [None, None, -2, _3Dconv1],
                                  name='3Dstack1', reg=self.reg)]
            for i in range(1, 3):
                _3Dstack.append(hourglass('3d', _3Dstack[-1][-1], [64, 64, 64, 32], [3, 3, 3, 3],
                                          [_3Dstack[-1][-2], None, _3Dstack[0][0], _3Dconv1], name='3Dstack%d' % (i+1),
                                          reg=self.reg))

            output_1 = conv_block(tf.layers.conv3d, _3Dstack[0][3], 32, 3, name='output_1_1', reg=self.reg)
            output_1 = conv_block(tf.layers.conv3d, output_1, 1, 3, name='output_1', reg=self.reg, apply_bn=False, apply_relu=False)
            outputs = [output_1]

            for i in range(1, 3):
                output = conv_block(tf.layers.conv3d, _3Dstack[i][3], 32, 3, name='output_%d_1' % (i+1), reg=self.reg)
                output = conv_block(tf.layers.conv3d, output, 1, 3, name='output_%d_2' % (i+1), reg=self.reg, apply_bn=False, apply_relu=False)
                output = tf.add(output, outputs[-1], name='output_%d' % (i+1))
                outputs.append(output)

        return outputs

    def output(self, outputs):
        disps = []
        for i, output in enumerate(outputs):
            squeeze = tf.squeeze(output, [4])
            transpose = tf.transpose(squeeze, [0, 2, 3, 1])
            upsample = tf.transpose(tf.image.resize_images(transpose, self.image_size_tf), [0, 3, 1, 2])
            disps.append(soft_arg_min(upsample, 'soft_arg_min_%d' % (i+1)))
        return disps

    def train(self, left, right, label, counter):
        _,loss = self.sess.run([self.train_op, self.loss],
            feed_dict={self.left: left, self.right: right, self.label:label,self.is_training:True})
		
        loss_line = self.sess.run(self.merge,
            feed_dict={self.left: left, self.right: right, self.label:label,self.is_training:True})
        self.writer.add_summary(loss_line, counter)		
        return loss
		
    def test(self, left, right, label):
        pred, loss = self.sess.run([self.disps, self.loss],
                                   feed_dict={self.left: left, self.right: right, self.label: label, self.is_training:True})
        return pred, loss

    def predict(self, left, right):
        pred = self.sess.run(self.disps,
                             feed_dict={self.left: left, self.right: right, self.is_training:True})
        return pred

if __name__ == '__main__':
    with tf.Session() as sess:
        model = Model(sess)
