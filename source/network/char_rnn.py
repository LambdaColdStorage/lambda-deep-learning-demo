import tensorflow as tf

rnn = tf.contrib.rnn


def net(inputs, rnn_size, num_rnn_layer, softmax_temprature,
        batch_size, vocab_size, is_training=True):

  with tf.variable_scope(name_or_scope='CharRNN',
                         values=[inputs],
                         reuse=tf.AUTO_REUSE):

    cell = rnn.MultiRNNCell([rnn.LSTMBlockCell(num_units=rnn_size)
                            for _ in range(num_rnn_layer)])

    c0 = tf.zeros([batch_size, rnn_size], tf.float32)
    h0 = tf.zeros([batch_size, rnn_size], tf.float32)
    c1 = tf.zeros([batch_size, rnn_size], tf.float32)
    h1 = tf.zeros([batch_size, rnn_size], tf.float32)

    initial = (rnn.LSTMStateTuple(c0, h0),
               rnn.LSTMStateTuple(c1, h1))

    embeddingW = tf.get_variable('embedding', [vocab_size, rnn_size])

    input_feature = tf.nn.embedding_lookup(embeddingW, inputs)

    input_list = tf.unstack(input_feature, axis=1)

    outputs, last_state = rnn.static_rnn(
      cell, input_list, initial, scope='rnnlm')
    last_state = tf.identity(last_state, 'last_state')

    output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])

    logits = tf.layers.dense(
      inputs=tf.layers.flatten(output),
      units=vocab_size,
      activation=tf.identity,
      use_bias=True,
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
      bias_initializer=tf.zeros_initializer())

    probabilities = tf.nn.softmax(logits / softmax_temprature, name='prob')

    return logits, probabilities