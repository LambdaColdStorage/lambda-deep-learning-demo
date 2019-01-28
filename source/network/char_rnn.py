import numpy as np

import tensorflow as tf

rnn = tf.contrib.rnn

# 28 is "T"
START_CHAR = 28
RNN_SIZE = 256
NUM_RNN_LAYER = 2
SOFTMAX_TEMPRATURE = 1.0


def net(x, feed_dict_seq, seq_length,
        batch_size, vocab_size, mode="train"):

  with tf.variable_scope(name_or_scope='CharRNN',
                         values=[x],
                         reuse=tf.AUTO_REUSE):

    if mode == "train" or mode == "eval":
      inputs = x
      c0 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      h0 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      c1 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      h1 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
    elif mode == "export":
      inputs = x[0]
      c0 = x[1]
      h0 = x[2]
      c1 = x[3]
      h1 = x[4]
      # inputs = tf.zeros([1, 1], tf.int32)
      # c0 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      # h0 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      # c1 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      # h1 = tf.zeros([batch_size, RNN_SIZE], tf.float32)      
    else:
      # Use placeholder in inference mode for both input and states
      # This allows taking the previous batch (step)'s output
      # as the input for the next batch.
      inputs = tf.placeholder(
        tf.int32,
        shape=(batch_size, seq_length),
        name="inputs")
      initial_value = np.array([[START_CHAR]], dtype=np.int32)
      feed_dict_seq[inputs] = initial_value

      c0 = tf.placeholder(
        tf.float32,
        shape=(batch_size, RNN_SIZE), name="c0")
      h0 = tf.placeholder(
        tf.float32,
        shape=(batch_size, RNN_SIZE), name="h0")
      c1 = tf.placeholder(
        tf.float32,
        shape=(batch_size, RNN_SIZE), name="c1")
      h1 = tf.placeholder(
        tf.float32,
        shape=(batch_size, RNN_SIZE), name="h1")

      initial_value = np.zeros(
        (batch_size, RNN_SIZE), dtype=float)
      feed_dict_seq[c0] = initial_value
      feed_dict_seq[h0] = initial_value
      feed_dict_seq[c1] = initial_value
      feed_dict_seq[h1] = initial_value

    initial_state = (rnn.LSTMStateTuple(c0, h0),
                     rnn.LSTMStateTuple(c1, h1))

    cell = rnn.MultiRNNCell([rnn.LSTMBlockCell(num_units=RNN_SIZE)
                            for _ in range(NUM_RNN_LAYER)])

    embeddingW = tf.get_variable('embedding', [vocab_size, RNN_SIZE])

    input_feature = tf.nn.embedding_lookup(embeddingW, inputs)

    input_list = tf.unstack(input_feature, axis=1)

    outputs, last_state = tf.nn.static_rnn(
      cell, input_list, initial_state)

    output = tf.reshape(tf.concat(outputs, 1), [-1, RNN_SIZE])

    logits = tf.layers.dense(
      inputs=tf.layers.flatten(output),
      units=vocab_size,
      activation=tf.identity,
      use_bias=True,
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
      bias_initializer=tf.zeros_initializer())

    probabilities = tf.nn.softmax(logits / SOFTMAX_TEMPRATURE, name='prob')

    return logits, probabilities, last_state, inputs
