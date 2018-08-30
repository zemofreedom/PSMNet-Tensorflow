import tensorflow as tf
import tensorflow.contrib as tfc


def conv_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None, reg=1e-4,
               apply_bn=True, apply_relu=True):
    with tf.variable_scope(name):
        conv_params = {
            'padding': 'same',
            'kernel_initializer': tfc.layers.xavier_initializer(),
            'kernel_regularizer': tfc.layers.l2_regularizer(reg),
            'bias_regularizer': tfc.layers.l2_regularizer(reg),
            'name': 'conv',
            'reuse': reuse
        }
        #这里需要注意，转置卷积不能加上空洞卷积的值
        if dilation_rate != -1:
            conv_params['dilation_rate'] = dilation_rate
        #if dilation_rate == -1:
        #    conv_params[]
        bottom = func(bottom, filters, kernel_size, strides, **conv_params)
        if apply_bn:
            bottom = tf.layers.batch_normalization(bottom,
                                                   training=tf.get_default_graph().get_tensor_by_name('is_training:0'),
                                                   reuse=reuse, name='bn')
        if apply_relu:
            bottom = tf.nn.relu(bottom, name='relu')
        return bottom


def res_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None, reg=1e-4,
              projection=False):
    with tf.variable_scope(name):
        short_cut = bottom
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, name='conv1', reuse=reuse,
                            reg=reg)
        bottom = conv_block(func, bottom, filters, kernel_size, 1, dilation_rate, name='conv2', reuse=reuse, reg=reg,
                            apply_relu=False)
        if projection:
            short_cut = tf.layers.conv2d(short_cut, filters, 1, strides, padding='same',
                                         kernel_initializer=tfc.layers.xavier_initializer(),
                                         kernel_regularizer=tfc.layers.l2_regularizer(reg),
                                         bias_regularizer=tfc.layers.l2_regularizer(reg),
                                         name='projection', reuse=reuse)
        bottom = tf.add(bottom, short_cut, 'add')
        return bottom


def SPP_branch(func, bottom, pool_size, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None,
               reg=1e-4, apply_bn=True, apply_relu=True):
    with tf.variable_scope(name):
        size = tf.shape(bottom)[1:3]
        bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool')
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, 'conv', reuse, reg,
                            apply_bn, apply_relu)
        bottom = tf.image.resize_images(bottom, size)
    return bottom


def hourglass(strs, bottom, filters_list, kernel_size_list, short_cut_list, dilation_rate=1, name=None, reg=1e-4):
    with tf.variable_scope(name):
        output = []
        conv_func, deconv_func = (tf.layers.conv2d, tf.layers.conv2d_transpose) if strs == '2d' else (tf.layers.conv3d, tf.layers.conv3d_transpose)
        #print(list(zip(filters_list, kernel_size_list, short_cut_list)))
        for i, (filters, kernel_size, short_cut) in enumerate(zip(filters_list, kernel_size_list, short_cut_list)):
            if i < len(filters_list) // 2:
                bottom = conv_block(conv_func, bottom, filters, kernel_size, strides=2, dilation_rate=dilation_rate,
                                    name='stack_%d_1' % (i+1), reg=reg)
                bottom = conv_block(conv_func, bottom, filters, kernel_size, dilation_rate=dilation_rate,
                                    name='stack_%d_2' % (i+1), reg=reg)
                if short_cut is not None:
                    if type(short_cut) is int:
                        short_cut = output[short_cut]
                    bottom = tf.add(bottom, short_cut, name='stack_%d' % (i+1))
            else:
                #反卷积有问题,必须确定batch-size,height,weight才能正确运行
                bottom = conv_block(deconv_func, bottom, filters, kernel_size, strides=2, dilation_rate=-1, name='stack_%d_1' % (i+1), reg=reg)
                #print('success')
                if short_cut is not None:
                    bottom = tf.add(bottom, short_cut, name='stack_%d' % (i + 1))
            output.append(bottom)
    return output


def soft_arg_min(filtered_cost_volume, name):
    with tf.variable_scope(name):
        #input.shape (batch, depth, H, W)
        # softargmin to disp image, outsize of (B, H, W)

        print('filtered_cost_volume:',filtered_cost_volume.shape)
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           dim=1, name='prob_volume')
        print('probability_volume:',probability_volume.shape)
        volume_shape = tf.shape(probability_volume)
        soft_1d = tf.cast(tf.range(0, volume_shape[1], dtype=tf.int32),tf.float32)
        soft_4d = tf.tile(soft_1d, tf.stack([volume_shape[0] * volume_shape[2] * volume_shape[3]]))
        soft_4d = tf.reshape(soft_4d, [volume_shape[0], volume_shape[2], volume_shape[3], volume_shape[1]])
        soft_4d = tf.transpose(soft_4d, [0, 3, 1, 2])
        estimated_disp_image = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        print(estimated_disp_image.shape)
        #estimated_disp_image = tf.expand_dims(estimated_disp_image, axis=3)
        return estimated_disp_image
