"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
# import importlib

import tensorflow as tf


class Modeler(object):
  def __init__(self, args, net):
    self.args = args
    self.net = net

    self.train_vars = []
    self.feed_dict_pre = {}
    self.feed_dict_seq = {}
    self.skip_l2_loss_vars = []

  def create_nonreplicated_fn(self, *argv):
    raise NotImplementedError()

  def model_fn(self, *argv):
    pass

  def create_graph_fn(self, *argv):
    pass

  def create_eval_metrics_fn(self, *argv):
    pass

  def create_loss_fn(self, *argv):
    pass

  def gether_train_vars(self):

    self.train_vars = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES)

    # Collect all trainale variables
    if self.args.trainable_var_list:
      self.train_vars = [v for v in self.train_vars
                         if any(x in v.name
                                for x in
                                self.args.trainable_var_list)]

    # Remove the blacklisted trainable variables
    if self.args.skip_trainable_var_list:
      self.train_vars = [v for v in self.train_vars
                         if not any(x in v.name
                                    for x in
                                    self.args.skip_trainable_var_list)]

  def create_optimizer(self, learning_rate):
    # Setup optimizer
    if self.args.optimizer == "adadelta":
      optimizer = tf.train.AdadeltaOptimizer(
          learning_rate=learning_rate)
    elif self.args.optimizer == "adagrad":
      optimizer = tf.train.AdagradOptimizer(
          learning_rate=learning_rate)
    elif self.args.optimizer == "adam":
      optimizer = tf.train.AdamOptimizer(
          learning_rate=learning_rate)
    elif self.args.optimizer == "ftrl":
      optimizer = tf.train.FtrlOptimizer(
          learning_rate=learning_rate)
    elif self.args.optimizer == "momentum":
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=0.9,
          name="Momentum")
    elif self.args.optimizer == "rmsprop":
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate=learning_rate)
    elif self.args.optimizer == "sgd":
      optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    else:
      raise ValueError("Optimizer [%s] was not recognized" %
                       self.args.optimizer)
    return optimizer

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
    # tf.summary.scalar("learning_rate", learning_rate)

    return learning_rate

  def l2_regularization(self):
    l2_var_list = [v for v in self.train_vars
                   if not any(x in v.name for
                              x in self.args.skip_l2_loss_vars)]

    loss_l2 = self.args.l2_weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in l2_var_list])
    return loss_l2

  def create_grad_fn(self, loss, clipping=None):
    self.optimizer = self.create_optimizer(self.learning_rate)
    grads = self.optimizer.compute_gradients(loss, var_list=self.train_vars)
    if clipping:
      grads = [(tf.clip_by_value(g, -clipping, clipping), v) for g, v in grads]
    return grads


def build(args, network, callbacks):
  return Modeler(args, network, callbacks)
