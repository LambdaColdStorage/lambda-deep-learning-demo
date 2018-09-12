"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import numpy as np

import tensorflow as tf

from modeler import Modeler

rnn = tf.contrib.rnn


class TextGenerationModeler(Modeler):
  def __init__(self, args, net, callbacks):
    super(TextGenerationModeler, self).__init__(args, net, callbacks)

    self.rnn_size = 256
    self.num_rnn_layer = 2
    self.grad_clip = 5.
    self.softmax_temprature = 1.0

  def get_dataset_info(self, inputter):
    self.seq_length = inputter.get_seq_length()
    self.num_samples = inputter.get_num_samples()
    self.vocab_size = inputter.get_vocab_size()
    self.chars = inputter.get_chars()

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_graph_fn(self, inputs, initial_state):
    is_training = (self.args.mode == "train")
    return self.net(inputs, initial_state, self.rnn_size, self.num_rnn_layer,
                    self.softmax_temprature, self.args.batch_size_per_gpu,
                    self.vocab_size, is_training=is_training)

  def create_eval_metrics_fn(self, logits, labels):
    classes = tf.argmax(logits, axis=1, output_type=tf.int32)
    equality = tf.equal(classes, tf.reshape(labels, [-1]))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy

  def create_loss_fn(self, logits, labels):
      loss_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=tf.reshape(labels, [-1])))

      loss_l2 = self.l2_regularization()

      loss = tf.identity(loss_cross_entropy + loss_l2, "total_loss")

      return loss

  def model_fn(self, x):

    # Input
    if self.args.mode == "train":
      inputs = x[0]
    else:
      inputs = tf.placeholder(
        tf.int32,
        shape=(self.args.batch_size_per_gpu, self.seq_length),
        name="inputs")

      initial_value = np.array([[5]], dtype=np.int32)
      self.feed_dict_seq = {inputs: initial_value}

    # States
    if self.args.mode == "train":
      c0 = tf.zeros([self.args.batch_size_per_gpu, self.rnn_size], tf.float32)
      h0 = tf.zeros([self.args.batch_size_per_gpu, self.rnn_size], tf.float32)
      c1 = tf.zeros([self.args.batch_size_per_gpu, self.rnn_size], tf.float32)
      h1 = tf.zeros([self.args.batch_size_per_gpu, self.rnn_size], tf.float32)
    else:
      c0 = tf.placeholder(
        tf.float32,
        shape=(self.args.batch_size_per_gpu, self.rnn_size), name="c0")
      h0 = tf.placeholder(
        tf.float32,
        shape=(self.args.batch_size_per_gpu, self.rnn_size), name="h0")
      c1 = tf.placeholder(
        tf.float32,
        shape=(self.args.batch_size_per_gpu, self.rnn_size), name="c1")
      h1 = tf.placeholder(
        tf.float32,
        shape=(self.args.batch_size_per_gpu, self.rnn_size), name="h1")

      initial_value = np.zeros(
        (self.args.batch_size_per_gpu, self.rnn_size), dtype=float)
      self.feed_dict_seq[c0] = initial_value
      self.feed_dict_seq[h0] = initial_value
      self.feed_dict_seq[c1] = initial_value
      self.feed_dict_seq[h1] = initial_value

    initial_state = (rnn.LSTMStateTuple(c0, h0),
                     rnn.LSTMStateTuple(c1, h1))

    logits, probabilities, last_state = self.create_graph_fn(inputs, initial_state)

    if self.args.mode == "train":
      labels = x[1]
      self.gether_train_vars()
      loss = self.create_loss_fn(logits, labels)
      grads = self.create_grad_fn(loss, self.grad_clip)
      accuracy = self.create_eval_metrics_fn(logits, labels)
      return {"loss": loss,
              "grads": grads,
              "accuracy": accuracy,
              "learning_rate": self.learning_rate}
    elif self.args.mode == "eval":
      pass
    elif self.args.mode == "infer":
      return {"inputs": inputs,
              "logits": logits,
              "probabilities": probabilities,
              "chars": tf.convert_to_tensor(self.chars),
              "last_state": last_state}


def build(args, net, callbacks):
  return TextGenerationModeler(args, net, callbacks)
