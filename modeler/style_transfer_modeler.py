"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib
import numpy as np

import tensorflow as tf

from modeler import Modeler


class StyleTransferModeler(Modeler):
  def __init__(self, args):
    super(StyleTransferModeler, self).__init__(args)
    self.net = getattr(importlib.import_module("network." + self.args.network),
                       "net")
    # self.class_names = self.args.class_names.split(",")
    # self.colors = np.random.randint(255,
    #                                 size=(self.args.num_classes, 3))

  def create_precomputation(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def model_fn(self, x):
    images = x[0]
    labels = x[1]
    logits, predictions = self.create_graph_fn(images)

    if self.args.mode == "train":
      loss = self.create_loss_fn(logits, labels)
      grads = self.create_grad_fn(loss)

    return {"loss": loss,
            "grads": grads}

  def create_graph_fn(self, input):
    is_training = (self.args.mode == "train")
    return self.net(input, self.args.num_classes,
                    is_training=is_training, data_format=self.args.data_format)

  def create_learning_rate_fn(self, global_step):
    """Create learning rate
    Returns:
      A learning rate calcualtor used by TF"s optimizer.
    """
    initial_learning_rate = self.args.learning_rate
    bs_per_gpu = self.args.batch_size_per_gpu
    num_gpu = self.args.num_gpu
    batches_per_epoch = (self.num_samples / (bs_per_gpu * num_gpu))
    boundaries = list(map(float,
                      self.args.piecewise_boundaries.split(",")))
    boundaries = [int(batches_per_epoch * boundary) for boundary in boundaries]

    decays = list(map(float,
                  self.args.piecewise_learning_rate_decay.split(",")))
    values = [initial_learning_rate * decay for decay in decays]

    learning_rate = tf.train.piecewise_constant(
      tf.cast(global_step, tf.int32), boundaries, values)

    tf.identity(learning_rate, name="learning_rate")
    tf.summary.scalar("learning_rate", learning_rate)

    return learning_rate

  def create_eval_metrics_fn(self, predictions, labels):
    pass

  def create_loss_fn(self, logits, labels):
    logits = tf.reshape(logits, [-1, self.args.num_classes])
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)

    loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

    l2_var_list = [v for v in tf.trainable_variables()]

    l2_var_list = [v for v in l2_var_list
                   if not any(x in v.name for
                              x in ["BatchNorm","preact","postnorm"])]

    loss_l2 = self.args.l2_weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in l2_var_list])

    loss = tf.identity(loss_cross_entropy + loss_l2, "total_loss")

    return loss

  def create_grad_fn(self, loss):
    self.optimizer = self.create_optimizer(self.learning_rate)
    grads = self.optimizer.compute_gradients(loss)

    return grads


def build(args):
  return StyleTransferModeler(args)
