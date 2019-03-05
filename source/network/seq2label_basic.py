import numpy as np

import tensorflow as tf

rnn = tf.contrib.rnn

EMBEDDING_SIZE = 200
NUM_RNN_LAYER = 2
RNN_SIZE = [128, 128]


def net(inputs, mask, num_classes, is_training, batch_size, vocab_size, embd=None, use_one_hot_embeddings=False):


  with tf.variable_scope(name_or_scope='seq2label_basic',
                         reuse=tf.AUTO_REUSE):

    initial_state = ()
    for i_layer in range(NUM_RNN_LAYER):
      initial_state = initial_state + \
        (rnn.LSTMStateTuple(tf.zeros([batch_size, RNN_SIZE[i_layer]], tf.float32),
                            tf.zeros([batch_size, RNN_SIZE[i_layer]], tf.float32)),)

    cell = rnn.MultiRNNCell([rnn.LSTMCell(num_units=RNN_SIZE[i_layer])
                            for i_layer in range(NUM_RNN_LAYER)])

    if len(embd) > 0:
      embeddingW = tf.get_variable(
        'embedding',
        initializer=tf.constant(embd),
        trainable=False)
    else:
      embeddingW = tf.get_variable(
      'embedding', [vocab_size, EMBEDDING_SIZE])

    # Only use the non-padded words
    sequence_length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

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