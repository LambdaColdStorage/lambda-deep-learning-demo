"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from modeler import Modeler

rnn = tf.contrib.rnn


class TextGenerationModeler(Modeler):
  def __init__(self, args):
    super(TextGenerationModeler, self).__init__(args)

    self.rnn_size = 256
    self.num_rnn_layer = 2
    self.grad_clip = 5.
    self.softmax_temprature = 1

    self.batch_size = (self.args.batch_size_per_gpu *
                       self.args.num_gpu)

    if self.args.mode == "train":
      self.create_callbacks(["train_basic", "train_loss",
                             "train_accuracy", "train_speed",
                             "train_summary"])
    elif self.args.mode == "eval":
      self.create_callbacks(["eval_basic", "eval_loss",
                             "eval_accuracy", "eval_speed",
                             "eval_summary"])
    elif self.args.mode == "infer":
      self.create_callbacks(["infer_basic",
                             "infer_display_char_rnn"])

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.vocab_size = inputter.get_vocab_size()

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_graph_fn(self, inputs):

    cell = rnn.MultiRNNCell([rnn.LSTMBlockCell(num_units=self.rnn_size)
                            for _ in range(self.num_rnn_layer)])

    def get_v(n):
        ret = tf.get_variable(n + '_unused', [self.batch_size, self.rnn_size],
                              trainable=False,
                              initializer=tf.constant_initializer())
        ret = tf.placeholder_with_default(
          ret, shape=[None, self.rnn_size], name=n)
        return ret

    initial = (rnn.LSTMStateTuple(get_v('c0'), get_v('h0')),
               rnn.LSTMStateTuple(get_v('c1'), get_v('h1')))

    embeddingW = tf.get_variable('embedding', [self.vocab_size, self.rnn_size])

    input_feature = tf.nn.embedding_lookup(embeddingW, inputs)

    input_list = tf.unstack(input_feature, axis=1)

    outputs, last_state = rnn.static_rnn(
      cell, input_list, initial, scope='rnnlm')
    last_state = tf.identity(last_state, 'last_state')

    output = tf.reshape(tf.concat(outputs, 1), [-1, self.rnn_size])

    logits = tf.layers.dense(
      inputs=tf.layers.flatten(output),
      units=self.vocab_size,
      activation=tf.identity,
      use_bias=True,
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
      bias_initializer=tf.zeros_initializer(),
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      reuse=tf.AUTO_REUSE)

    return logits

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

    inputs = x[0]
    labels = x[1]

    logits = self.create_graph_fn(inputs)

    if self.args.mode == "train":
      self.gether_train_vars()
      loss = self.create_loss_fn(logits, labels)
      grads = self.create_grad_fn(loss)
      accuracy = self.create_eval_metrics_fn(logits, labels)
      return {"loss": loss,
              "grads": grads,
              "accuracy": accuracy,
              "learning_rate": self.learning_rate}
    elif self.args.mode == "eval":
      pass
    elif self.args.mode == "infer":
      pass


def build(args):
  return TextGenerationModeler(args)
