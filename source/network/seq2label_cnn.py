import numpy as np

import tensorflow as tf

EMBEDDING_SIZE = 16

def net(inputs, mask, num_classes, is_training, batch_size, vocab_size, embd=None, use_one_hot_embeddings=False):


  with tf.variable_scope(name_or_scope='seq2label_cnn',
                         reuse=tf.AUTO_REUSE):

    if len(embd) > 0:
      embeddingW = tf.get_variable(
        'embedding',
        initializer=tf.constant(embd),
        trainable=True)
    else:
      embeddingW = tf.get_variable(
      'embedding', [vocab_size, EMBEDDING_SIZE])

    # Only use the non-padded words
    sequence_length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

    input_feature = tf.nn.embedding_lookup(embeddingW, inputs)

    output = tf.reduce_mean(input_feature, axis=[2])

    output = tf.layers.dense(
      inputs=output,
      units=16,
      activation=tf.nn.relu,
      use_bias=True,
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
      bias_initializer=tf.zeros_initializer())

    logits = tf.layers.dense(
      inputs=output,
      units=2,
      activation=tf.identity,
      use_bias=True,
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
      bias_initializer=tf.zeros_initializer())

    probabilities = tf.nn.softmax(logits, name='prob')

    return logits, probabilities