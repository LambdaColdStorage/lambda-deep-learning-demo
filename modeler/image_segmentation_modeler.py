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
    self.feed_dict_ops = {}

    if self.args.mode == "train":
      self.create_callbacks(["train_basic", "train_loss",
                             "train_accuracy", "train_speed"])
    elif self.args.mode == "eval":
      self.create_callbacks(["eval_basic", "eval_accuracy", "eval_speed"])
    elif self.args.mode == "infer":
      self.create_callbacks(["infer_basic",
                             "infer_display_image_segmentation"])

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_graph_fn(self, input):
    is_training = (self.args.mode == "train")
    return self.net(input, self.args.num_classes,
                    is_training=is_training, data_format=self.args.data_format)

  def create_eval_metrics_fn(self, predictions, labels):
    equality = tf.equal(predictions["classes"],
                        labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy

  def create_loss_fn(self, logits, labels):
    logits = tf.reshape(logits, [-1, self.args.num_classes])
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)

    loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

    loss_l2 = self.l2_regularization()

    loss = tf.identity(loss_cross_entropy + loss_l2, "total_loss")

    return loss

  def model_fn(self, x):
    images = x[0]
    labels = x[1]
    logits, predictions = self.create_graph_fn(images)

    if self.args.mode == "train":
      self.gether_train_vars()
      loss = self.create_loss_fn(logits, labels)
      grads = self.create_grad_fn(loss)
      accuracy = self.create_eval_metrics_fn(
        predictions, labels)
      return {"loss": loss,
              "grads": grads,
              "accuracy": accuracy}
    elif self.args.mode == "eval":
      self.gether_train_vars()
      loss = self.create_loss_fn(logits, labels)
      accuracy = self.create_eval_metrics_fn(
        predictions, labels)
      return {"loss": loss,
              "accuracy": accuracy}
    elif self.args.mode == "infer":
      return {"classes": predictions["classes"],
              "probabilities": predictions["probabilities"]}


def build(args):
  return ImageSegmentationModeler(args)
