import numpy as np

import tensorflow as tf

rnn = tf.contrib.rnn

RNN_SIZE = 256
NUM_RNN_LAYER = 2


def length(sequence):
  # Measure sentence length by skipping the padded words (-1)
  used = tf.to_float(tf.math.greater_equal(sequence, 0))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length


def net(x, batch_size, vocab_size, mode="train"):

  with tf.variable_scope(name_or_scope='seq2label',
                         values=[x],
                         reuse=tf.AUTO_REUSE):

    if mode == "train" or mode == "eval" or mode == 'infer':
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

    initial_state = (rnn.LSTMStateTuple(c0, h0),
                     rnn.LSTMStateTuple(c1, h1))

    cell = rnn.MultiRNNCell([rnn.LSTMCell(num_units=RNN_SIZE)
                            for _ in range(NUM_RNN_LAYER)])

    embeddingW = tf.get_variable('embedding', [vocab_size, RNN_SIZE])

    # Hack: use only the non-padded words
    sequence_length = length(inputs)

    inputs = inputs + tf.cast(tf.math.equal(inputs, -1), tf.int32)

    input_feature = tf.nn.embedding_lookup(embeddingW, inputs)

    output, _ = tf.nn.dynamic_rnn(
      cell,
      input_feature,
      initial_state=initial_state,
      sequence_length=sequence_length)

    # The last output is the encoding of the entire sentence
    idx_gather = tf.concat(
      [tf.expand_dims(tf.range(tf.shape(output)[0], delta=1), axis=1),
       tf.expand_dims(sequence_length - 1, axis=1)], axis=1)

    last_output = tf.gather_nd(output, indices=idx_gather)

    logits = tf.layers.dense(
      inputs=last_output,
      units=2,
      activation=tf.identity,
      use_bias=True,
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
      bias_initializer=tf.zeros_initializer())

    probabilities = tf.nn.softmax(logits, name='prob')

    return logits, probabilities