import tensorflow as tf

BASE_FILTER = 64
# DATA_FORMAT = 'channels_first'


def net(inputs, num_classes,
        is_training=True, data_format="channels_first"):
  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  with tf.variable_scope(name_or_scope='FCN',
                         values=[inputs],
                         reuse=tf.AUTO_REUSE):
    kernel_init = tf.variance_scaling_initializer()
    # Encoder
    inputs = tf.layers.conv2d(inputs=inputs,
                              filters=BASE_FILTER,
                              kernel_size=[4, 4],
                              strides=(2, 2),
                              padding=('SAME'),
                              data_format=data_format,
                              activation=tf.nn.relu,
                              kernel_initializer=kernel_init,
                              name='encode1')

    inputs = tf.layers.conv2d(inputs=inputs,
                              filters=BASE_FILTER * 2,
                              kernel_size=[4, 4],
                              strides=(2, 2),
                              padding=('SAME'),
                              data_format=data_format,
                              activation=tf.nn.relu,
                              kernel_initializer=kernel_init,
                              name='encode2')

    inputs = tf.layers.conv2d(inputs=inputs,
                              filters=BASE_FILTER * 4,
                              kernel_size=[4, 4],
                              strides=(2, 2),
                              padding='SAME',
                              data_format=data_format,
                              activation=tf.nn.relu,
                              kernel_initializer=kernel_init,
                              name='encode3')

    inputs = tf.layers.conv2d(inputs=inputs,
                              filters=BASE_FILTER * 8,
                              kernel_size=[4, 4],
                              strides=(2, 2),
                              padding='SAME',
                              data_format=data_format,
                              activation=tf.nn.relu,
                              kernel_initializer=kernel_init,
                              name='encode4')

    # Decoder
    inputs = tf.layers.conv2d_transpose(inputs=inputs,
                                        filters=BASE_FILTER * 4,
                                        kernel_size=[4, 4],
                                        strides=(2, 2),
                                        padding='SAME',
                                        data_format=data_format,
                                        activation=tf.nn.relu,
                                        kernel_initializer=kernel_init,
                                        name='decode4')

    inputs = tf.layers.conv2d_transpose(inputs=inputs,
                                        filters=BASE_FILTER * 2,
                                        kernel_size=[4, 4],
                                        strides=(2, 2),
                                        padding='SAME',
                                        data_format=data_format,
                                        activation=tf.nn.relu,
                                        kernel_initializer=kernel_init,
                                        name='decode3')

    inputs = tf.layers.conv2d_transpose(inputs=inputs,
                                        filters=BASE_FILTER,
                                        kernel_size=[4, 4],
                                        strides=(2, 2),
                                        padding='SAME',
                                        data_format=data_format,
                                        activation=tf.nn.relu,
                                        kernel_initializer=kernel_init,
                                        name='decode2')

    inputs = tf.layers.conv2d_transpose(inputs=inputs,
                                        filters=num_classes,
                                        kernel_size=[4, 4],
                                        strides=(2, 2),
                                        padding='SAME',
                                        data_format=data_format,
                                        activation=None,
                                        kernel_initializer=kernel_init,
                                        name='decode1')

    # Always use channels_last so we can have a canonical loss implementation
    if data_format == 'channels_first':
      inputs = tf.transpose(inputs, [0, 2, 3, 1])

    logits = tf.identity(inputs)

    predictions = {
      'classes': tf.argmax(logits, axis=3),
      'probabilities': tf.nn.softmax(logits, dim=3, name='softmax_tensor')}

    return logits, predictions
