"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import abc
import six

import tensorflow as tf

@six.add_metaclass(abc.ABCMeta)
class Modeler(object):
  def __init__(self, args):
    self.args = args

  @abc.abstractmethod
  def create_precomputation(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def model_fn(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_graph_fn(self, *argv):
    raise NotImplementedError

  @abc.abstractmethod
  def create_eval_metrics_fn(self, *argv):
    raise NotImplementedError

  @abc.abstractmethod
  def create_loss_fn(self, *argv):
    raise NotImplementedError

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

def build(args):
  return Modeler(args)
