import numpy as np
import pickle

import tensorflow as tf

VGG_PARAMS_FILE = "/home/ubuntu/git/caffe_ssd/SSD_512x512.p"

def vgg_block(outputs, params, name, data_format, num_conv):

    for i in range(num_conv):
        layer_name = name + "_" + str(i + 1)
        w = np.swapaxes(np.swapaxes(np.swapaxes(params[layer_name][0], 0, 3), 1, 2), 0, 1)
        b = params[layer_name][1]
        outputs = tf.layers.conv2d(
                outputs,
    	    filters=w.shape[3],
    	    kernel_size=(w.shape[0], w.shape[1]),
    	    strides=(1, 1),
    	    padding=("SAME"),
    	    data_format=data_format,
    	    kernel_initializer=tf.constant_initializer(w),
    	    bias_initializer=tf.constant_initializer(b),
    	    activation=tf.nn.relu,
    	    name=layer_name)
    return outputs

def vgg_pool(outputs, params, name, data_format, pool_sz=2, pool_stride=2):
    layer_name = "pool" + name[-1]
    outputs = tf.layers.max_pooling2d(
    	outputs,
    	pool_size=(pool_sz, pool_sz),
    	strides=(pool_stride, pool_stride),
    	padding="SAME",
    	data_format=data_format,
    	name=layer_name)

    return outputs

def vgg_mod(outputs, params, name, data_format, dilation=1):
    w = np.swapaxes(np.swapaxes(np.swapaxes(params[name][0], 0, 3), 1, 2), 0, 1)
    b = params[name][1]
    outputs = tf.layers.conv2d(
            outputs,
        filters=w.shape[3],
        kernel_size=(w.shape[0], w.shape[1]),
        strides=(1, 1),
        padding=("SAME"),
        data_format=data_format,
        dilation_rate=(dilation, dilation),
        kernel_initializer=tf.constant_initializer(w),
        bias_initializer=tf.constant_initializer(b),
        activation=tf.nn.relu,
        name=name)
    return outputs

def vgg(outputs, params, data_format):
	outputs = vgg_block(outputs, params, "conv1", data_format, num_conv=2)
        outputs = vgg_pool(outputs, params, "pool1", data_format)
	outputs = vgg_block(outputs, params, "conv2", data_format, num_conv=2)
        outputs = vgg_pool(outputs, params, "pool2",data_format)
	outputs = vgg_block(outputs, params, "conv3", data_format, num_conv=3)
        outputs = vgg_pool(outputs, params, "pool3", data_format)
        outputs_conv4_3 = vgg_block(outputs, params, "conv4", data_format, num_conv=3)
        outputs = vgg_pool(outputs_conv4_3, params, "pool4", data_format)
        outputs = vgg_block(outputs, params, "conv5", data_format, num_conv=3)
        outputs = vgg_pool(outputs, params, "pool5", data_format, pool_sz=3, pool_stride=1)
        outputs = vgg_mod(outputs, params, "fc6", data_format, dilation=6)
        outputs_fc7 = vgg_mod(outputs, params, "fc7", data_format)

	return [outputs_conv4_3, outputs_fc7]


def net(inputs, data_format):

    params = pickle.load(open(VGG_PARAMS_FILE, "rb"))

    with tf.variable_scope(name_or_scope='VGG',
                           values=[inputs],
                           reuse=tf.AUTO_REUSE):
        outputs = vgg(inputs, params, data_format)

        return outputs
