import tensorflow as tf

rnn = tf.contrib.rnn


def net(inputs, rnn_size, num_rnn_layer, softmax_temprature,
        batch_size, vocab_size, is_training=True):
  cell = rnn.MultiRNNCell([rnn.LSTMBlockCell(num_units=rnn_size)
                          for _ in range(num_rnn_layer)])

  def get_v(n):
      ret = tf.get_variable(n + '_unused', [batch_size, rnn_size],
                            trainable=False,
                            initializer=tf.constant_initializer())
      ret = tf.placeholder_with_default(
        ret, shape=[None, rnn_size], name=n)
      return ret

  initial = (rnn.LSTMStateTuple(get_v('c0'), get_v('h0')),
             rnn.LSTMStateTuple(get_v('c1'), get_v('h1')))

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
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    reuse=tf.AUTO_REUSE)

  probabilities = tf.nn.softmax(logits / softmax_temprature, name='prob')

  return logits, probabilities
