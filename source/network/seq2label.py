import numpy as np

import tensorflow as tf

rnn = tf.contrib.rnn

START_CHAR = 9
RNN_SIZE = 256
NUM_RNN_LAYER = 2
SOFTMAX_TEMPRATURE = 1.0


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
      # inputs = tf.zeros([1, 1], tf.int32)
      # c0 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      # h0 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      # c1 = tf.zeros([batch_size, RNN_SIZE], tf.float32)
      # h1 = tf.zeros([batch_size, RNN_SIZE], tf.float32)      
    # else:
    #   # Use placeholder in inference mode for both input and states
    #   # This allows taking the previous batch (step)'s output
    #   # as the input for the next batch.
    #   inputs = tf.placeholder(
    #     tf.int32,
    #     shape=(batch_size, seq_length),
    #     name="inputs")
    #   initial_value = np.array([[START_CHAR]], dtype=np.int32)
    #   feed_dict_seq[inputs] = initial_value

    #   c0 = tf.placeholder(
    #     tf.float32,
    #     shape=(batch_size, RNN_SIZE), name="c0")
    #   h0 = tf.placeholder(
    #     tf.float32,
    #     shape=(batch_size, RNN_SIZE), name="h0")
    #   c1 = tf.placeholder(
    #     tf.float32,
    #     shape=(batch_size, RNN_SIZE), name="c1")
    #   h1 = tf.placeholder(
    #     tf.float32,
    #     shape=(batch_size, RNN_SIZE), name="h1")

    #   initial_value = np.zeros(
    #     (batch_size, RNN_SIZE), dtype=float)
    #   feed_dict_seq[c0] = initial_value
    #   feed_dict_seq[h0] = initial_value
    #   feed_dict_seq[c1] = initial_value
    #   feed_dict_seq[h1] = initial_value

    initial_state = (rnn.LSTMStateTuple(c0, h0),
                     rnn.LSTMStateTuple(c1, h1))

    cell = rnn.MultiRNNCell([rnn.LSTMCell(num_units=RNN_SIZE)
                            for _ in range(NUM_RNN_LAYER)])

    embeddingW = tf.get_variable('embedding', [vocab_size, RNN_SIZE])

    input_feature = tf.nn.embedding_lookup(embeddingW, inputs)

    output, _ = tf.nn.dynamic_rnn(
      cell,
      input_feature,
      initial_state=initial_state)

    # The last output is the encoding of the entire sentence
    last_output = tf.gather(output, indices=tf.shape(output)[1], axis=1)

    logits = tf.layers.dense(
      inputs=last_output,
      units=2,
      activation=tf.identity,
      use_bias=True,
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
      bias_initializer=tf.zeros_initializer())

    probabilities = tf.nn.softmax(logits, name='prob')

    return logits, probabilities