import tensorflow as tf

para_conv = [{'filters': 32,
              'kernel_size': [8, 8],
              'strides': [1, 1]},
             {'filters': 64,
              'kernel_size': [4, 4],
              'strides': [2, 2]},
             {'filters': 128,
              'kernel_size': [4, 4],
              'strides': [2, 2]}]

para_deconv = [{'filters': 64,
                'kernel_size': [4, 4],
                'strides': [2, 2]},
               {'filters': 32,
                'kernel_size': [4, 4],
                'strides': [2, 2]},
               {'filters': 3,
                'kernel_size': [8, 8],
                'strides': [1, 1]}]

para_res = {'filters': 128,
            'kernel_size': [4, 4],
            'strides': [1, 1]}


def conv_layer(inputs, para, name, data_format):
  outputs = tf.layers.conv2d(inputs=inputs,
                             filters=para['filters'],
                             kernel_size=para['kernel_size'],
                             strides=para['strides'],
                             padding='SAME',
                             data_format=data_format,
                             name=name,
                             reuse=tf.AUTO_REUSE)
  return outputs


def deconv_layer(inputs, para, name, data_format):
  outputs = tf.layers.conv2d_transpose(inputs=inputs,
                                       filters=para['filters'],
                                       kernel_size=para['kernel_size'],
                                       strides=para['strides'],
                                       padding='SAME',
                                       data_format=data_format,
                                       name=name,
                                       reuse=tf.AUTO_REUSE)
  return outputs


def instance_norm_layer(inputs, data_format):
  if data_format == 'channels_last':
    batch, rows, cols, channels = [i.value for i in inputs.get_shape()]
    mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)
  else:
    batch, channels, rows, cols = [i.value for i in inputs.get_shape()]
    mu, sigma_sq = tf.nn.moments(inputs, [2, 3], keep_dims=True)

  epsilon = 1e-3
  outputs = (inputs - mu) / (sigma_sq + epsilon)**(.5)
  return outputs


def residual_layer(inputs, para, name, data_format):
  outputs = tf.nn.relu(instance_norm_layer(
                       conv_layer(inputs, para, name=name,
                                  data_format=data_format),
                       data_format=data_format))
  outputs = instance_norm_layer(conv_layer(outputs, para, name=name,
                                           data_format=data_format),
                                data_format=data_format)
  return inputs + outputs


def net(inputs, data_format):
  outputs = (tf.transpose(inputs, [0, 3, 1, 2])
             if data_format == 'channels_first'
             else inputs)
  outputs = tf.nn.relu(instance_norm_layer(
                       conv_layer(outputs, para_conv[0],
                                  name="encode_1", data_format=data_format),
                       data_format))
  outputs = tf.nn.relu(instance_norm_layer(
                       conv_layer(outputs, para_conv[1],
                                  name="encode_2", data_format=data_format),
                       data_format))
  outputs = tf.nn.relu(instance_norm_layer(
                       conv_layer(outputs, para_conv[2],
                                  name="encode_3", data_format=data_format),
                       data_format))
  outputs = residual_layer(outputs, para_res,
                           name="resisdual_1", data_format=data_format)
  outputs = residual_layer(outputs, para_res,
                           name="resisdual_2", data_format=data_format)
  outputs = residual_layer(outputs, para_res,
                           name="resisdual_3", data_format=data_format)
  outputs = residual_layer(outputs, para_res,
                           name="resisdual_4", data_format=data_format)
  outputs = residual_layer(outputs, para_res,
                           name="resisdual_5", data_format=data_format)
  outputs = tf.nn.relu(instance_norm_layer(
                       deconv_layer(outputs, para_deconv[0],
                                    name="decode_1", data_format=data_format),
                       data_format=data_format))
  outputs = tf.nn.relu(instance_norm_layer(
                       deconv_layer(outputs, para_deconv[1],
                                    name="decode_2", data_format=data_format),
                       data_format=data_format))
  outputs = instance_norm_layer(conv_layer(
                                outputs, para_deconv[2],
                                name="decode_3", data_format=data_format),
                                data_format=data_format)
  outputs = tf.nn.tanh(outputs) * 127.5 + 255. / 2

  outputs = (tf.transpose(outputs, [0, 2, 3, 1])
             if data_format == 'channels_first'
             else outputs)

  return outputs
