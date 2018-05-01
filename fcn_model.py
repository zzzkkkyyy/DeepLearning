import tensorflow as tf
import numpy as np
from math import ceil

input_width = 256
input_height = 256
        

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def batch_norm_layer(x, train_phase, scope_bn):
    shape = x.get_shape().as_list()
    x_unrolled = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape = [x_unrolled.shape[-1]]), name = 'beta', trainable = True)
        gamma = tf.Variable(tf.constant(1.0, shape = [x_unrolled.shape[-1]]), name = 'gamma', trainable = True)
        batch_mean, batch_var = tf.nn.moments(x_unrolled, axes = [0], name = 'moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x_unrolled, mean, var, beta, gamma, 1e-3)
        normed = tf.reshape(normed, tf.shape(x))
    return normed

def conv_layer(input, filter_shape, stride, batch = True):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(input, filter = filter_, strides = [1, stride, stride, 1], padding = 'SAME')
    mean, var = tf.nn.moments(conv, [0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]))
    gamma = weight_variable([out_channels])
    if batch is True:
        conv = tf.nn.batch_norm_with_global_normalization(conv, mean, var, beta, 
                                                                gamma, 0.001, scale_after_normalization = True)
    out = tf.nn.relu(conv)
    return out

def max_pool_layer(input, size, stride):
    return tf.nn.max_pool(input, ksize = [1, size, size, 1], strides = [1, stride, stride, 1], padding = 'SAME')

def residual_block(input, output_depth, down_sample):
    input_depth = input.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1, 3, 3, 1]
        input = tf.nn.max_pool(input, ksize = filter_, strides = filter_, padding='SAME')
    conv1 = conv_layer(input, [1, 1, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)
    conv3 = conv_layer(conv2, [1, 1, output_depth, output_depth * 4], 1)
    if input_depth != output_depth * 4:
        input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_depth * 4 - input_depth]])
    else:
        input_layer = input
    res = conv3 + input_layer
    return res

def unpooling_layer(input, name = 'unpooling'):
    with tf.name_scope(name) as scope:
        shape = input.get_shape().as_list()
        dim = len(shape[1:-1])
        out = (tf.reshape(input, [-1] + shape[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in shape[1:-1]] + [shape[-1]]
        out = tf.reshape(out, out_size, name = scope)
    return out

def up_proj(input, kernel_0, kernel, stride = 1):
    max_pool_0 = unpooling_layer(input)
    output_0 = conv_layer(max_pool_0, kernel_0, stride, batch = False)
    
    conv_x = conv_layer(max_pool_0, kernel[0], stride, batch = False)
    conv_x = tf.nn.relu(conv_x)
    conv_x = conv_layer(conv_x, kernel[1], stride, batch = False)
    return tf.nn.relu(conv_x + output_0)

def construct_layer(input, is_training):
    with tf.variable_scope('conv1'):
        conv1 = conv_layer(input, [7, 7, 3, 64], 1)
    conv2 = max_pool_layer(conv1, 3, 2)
    with tf.variable_scope('conv2'):
        for i in range(3):
            conv2 = residual_block(conv2, 64, False)    
    conv3 = max_pool_layer(conv2, 3, 2)
    with tf.variable_scope('conv3'):
        for i in range(4):
            conv3 = residual_block(conv3, 128, False)
    conv4 = max_pool_layer(conv3, 3, 2)
    with tf.variable_scope('conv4'):
        for i in range(6):
            conv4 = residual_block(conv4, 256, False)
    conv5 = max_pool_layer(conv4, 3, 2)
    with tf.variable_scope('conv5'):
        for i in range(3):
            conv5 = residual_block(conv5, 512, False)
    conv6 = max_pool_layer(conv5, 3, 2)
    with tf.variable_scope('conv6'):
        channels = conv6.get_shape().as_list()[3]
        conv6 = conv_layer(conv6, [1, 1, channels, channels // 2], 1)
    up_proj_x = conv6
    #conv7 = conv_layer(conv6, [1, 1, channels // 4, channels // 4], 1)
    #conv_conv = unpooling_layer(up_proj_x)
    #conv_conv = conv_layer(conv_conv, [1, 1, channels // 4, channels // 4], 1)
    epoch_size = 512
    for i in range(4):
        with tf.variable_scope('deconv{}'.format(i + 1)):
            kernel_0 = [5, 5, epoch_size * 2, epoch_size]
            kernel = []
            kernel.append([5, 5, epoch_size * 2, epoch_size])
            kernel.append([3, 3, epoch_size, epoch_size])
            epoch_size = epoch_size // 2
            up_proj_x = up_proj(up_proj_x, kernel_0, kernel, 1)
    decode_1 = up_proj_x
    conv7 = conv_layer(decode_1, [3, 3, 64, 151], 1)
    #result = unpooling_layer(conv7)
    #result = conv_layer(result, [3, 3, 1, 1], 1)
    #result = tf.image.resize_images(conv7, tf.Variable([input_width * 2, input_height]))
    result = tf.image.resize_bilinear(conv7, tf.Variable([2 * conv7.get_shape().as_list()[1], 2 * conv7.get_shape().as_list()[2]]))
    #result = tf.image.resize_bilinear(conv7, tf.convert_to_tensor(np.array([2 * conv7.get_shape().as_list()[1], 2 * conv7.get_shape().as_list()[2]])))
    prediction = tf.argmax(result, dimension = 3, name = "prediction")
    return tf.expand_dims(prediction, dim = 3), result
    #return result
        
        
        
        
        
