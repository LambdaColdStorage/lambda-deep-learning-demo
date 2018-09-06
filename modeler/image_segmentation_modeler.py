"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib
import numpy as np

import tensorflow as tf

from modeler import Modeler


class ImageSegmentationModeler(Modeler):
  def __init__(self, args):
    super(ImageSegmentationModeler, self).__init__(args)
    self.net = getattr(importlib.import_module("network." + self.args.network),
                       "net")
    self.class_names = self.args.class_names.split(",")
    self.colors = np.random.randint(255,
                                    size=(self.args.num_classes, 3))
    self.train_skip_vars = []
    self.l2_loss_skip_vars = ["BatchNorm", "preact", "postnorm"]
    self.train_vars = []
    self.pre_compute_ops = {}

  def create_precomputation(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def model_fn(self, x):
    images = x[0]
    labels = x[1]
    logits, predictions = self.create_graph_fn(images)

    self.gether_train_vars()

    if self.args.mode == "train":
      loss = self.create_loss_fn(logits, labels)
      grads = self.create_grad_fn(loss)

    return {"loss": loss,
            "grads": grads}

  def create_graph_fn(self, input):
    is_training = (self.args.mode == "train")
    return self.net(input, self.args.num_classes,
                    is_training=is_training, data_format=self.args.data_format)

  def create_eval_metrics_fn(self, predictions, labels):
    pass

  def create_loss_fn(self, logits, labels):
    logits = tf.reshape(logits, [-1, self.args.num_classes])
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)

    loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

    loss_l2 = self.l2_regularization()

    loss = tf.identity(loss_cross_entropy + loss_l2, "total_loss")

    return loss


def build(args):
  return ImageSegmentationModeler(args)
