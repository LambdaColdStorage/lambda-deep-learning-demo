import tensorflow as tf

BASE_FILTER = 32


def merge(encode, decode, depth, data_format):
  encode_shape = tf.shape(encode)
  decode_shape = tf.shape(decode)

  # offsets for the top left corner of the crop
  if data_format == 'channels_first':
    offsets = [0, 0, (decode_shape[2] - encode_shape[2]) // 2,
               (decode_shape[3] - encode_shape[3]) // 2]
    size = [-1, depth, encode_shape[2], encode_shape[3]]
    decode_crop = tf.slice(decode, offsets, size)
    output = tf.concat([encode, decode_crop], 1)
  else:
    offsets = [0, (decode_shape[1] - encode_shape[1]) // 2,
               (decode_shape[2] - encode_shape[2]) // 2, 0]
    size = [-1, encode_shape[1], encode_shape[2], depth]
    decode_crop = tf.slice(decode, offsets, size)
    output = tf.concat([encode, decode_crop], 3)

  return output


def net(inputs, num_classes, is_training=True,
        data_format="channels_first"):
  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  with tf.variable_scope(name_or_scope='UNET',
                         values=[inputs],
                         reuse=tf.AUTO_REUSE):

    kernel_init = tf.variance_scaling_initializer()

    # Encoder
    encoder1 = tf.layers.conv2d(inputs=inputs,
                                filters=BASE_FILTER,
                                kernel_size=[4, 4],
                                strides=(2, 2),
                                padding=('SAME'),
                                data_format=data_format,
                                activation=tf.nn.relu,
                                kernel_initializer=kernel_init,
                                name='encoder1')

    encoder2 = tf.layers.conv2d(inputs=encoder1,
                                filters=BASE_FILTER * 2,
                                kernel_size=[4, 4],
                                strides=(2, 2),
                                padding=('SAME'),
                                data_format=data_format,
                                activation=tf.nn.relu,
                                kernel_initializer=kernel_init,
                                name='encoder2')

    encoder3 = tf.layers.conv2d(inputs=encoder2,
                                filters=BASE_FILTER * 4,
                                kernel_size=[4, 4],
                                strides=(2, 2),
                                padding='SAME',
                                data_format=data_format,
                                activation=tf.nn.relu,
                                kernel_initializer=kernel_init,
                                name='encoder3')

    encoder4 = tf.layers.conv2d(inputs=encoder3,
                                filters=BASE_FILTER * 8,
                                kernel_size=[4, 4],
                                strides=(2, 2),
                                padding='SAME',
                                data_format=data_format,
                                activation=tf.nn.relu,
                                kernel_initializer=kernel_init,
                                name='encoder4')

    # Decoder
    decoder4 = tf.layers.conv2d_transpose(inputs=encoder4,
                                          filters=BASE_FILTER * 4,
                                          kernel_size=[4, 4],
                                          strides=(2, 2),
                                          padding='SAME',
                                          data_format=data_format,
                                          activation=tf.nn.relu,
                                          kernel_initializer=kernel_init,
                                          name='decoder4')
    decoder4 = merge(encoder3, decoder4, BASE_FILTER * 4, data_format)

    decoder3 = tf.layers.conv2d_transpose(inputs=decoder4,
                                          filters=BASE_FILTER * 2,
                                          kernel_size=[4, 4],
                                          strides=(2, 2),
                                          padding='SAME',
                                          data_format=data_format,
                                          activation=tf.nn.relu,
                                          kernel_initializer=kernel_init,
                                          name='decoder3')
    decoder3 = merge(encoder2, decoder3, BASE_FILTER * 2, data_format)

    decoder2 = tf.layers.conv2d_transpose(inputs=decoder3,
                                          filters=BASE_FILTER,
                                          kernel_size=[4, 4],
                                          strides=(2, 2),
                                          padding='SAME',
                                          data_format=data_format,
                                          activation=tf.nn.relu,
                                          kernel_initializer=kernel_init,
                                          name='decoder2')
    decoder2 = merge(encoder1, decoder2, BASE_FILTER, data_format)

    decoder1 = tf.layers.conv2d_transpose(inputs=decoder2,
                                          filters=num_classes,
                                          kernel_size=[4, 4],
                                          strides=(2, 2),
                                          padding='SAME',
                                          data_format=data_format,
                                          activation=None,
                                          kernel_initializer=kernel_init,
                                          name='decoder1')

    # Always use channels_last so we can have a canonical loss implementation
    if data_format == 'channels_first':
      decoder1 = tf.transpose(decoder1, [0, 2, 3, 1])

    logits = tf.identity(decoder1)

    predictions = {
      'classes': tf.argmax(logits, axis=3),
      'probabilities': tf.nn.softmax(logits, dim=3, name='softmax_tensor')}
    return logits, predictions
