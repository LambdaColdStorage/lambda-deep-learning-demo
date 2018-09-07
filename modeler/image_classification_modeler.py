"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib

import tensorflow as tf

from modeler import Modeler


class ImageClassificationModeler(Modeler):
  def __init__(self, args):
    super(ImageClassificationModeler, self).__init__(args)
    self.net = getattr(importlib.import_module("network." + self.args.network),
                       "net")
    self.train_skip_vars = []
    self.l2_loss_skip_vars = ["BatchNorm", "preact", "postnorm"]
    self.train_vars = []
    self.feed_dict_ops = {}

    if self.args.mode == "train":
      self.create_callbacks(["train_basic", "train_loss", "train_accuracy", "train_speed"])
    elif self.args.mode == "eval":
      self.create_callbacks(["eval_basic", "eval_accuracy", "eval_speed"])

  def create_nonreplicated_fn(self):
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
      accuracy = self.create_eval_metrics_fn(
        predictions, labels)
      return {"loss": loss,
              "grads": grads,
              "accuracy": accuracy}
    elif self.args.mode == "eval":
      loss = self.create_loss_fn(logits, labels)
      accuracy = self.create_eval_metrics_fn(
        predictions, labels)
      return {"loss": loss,
              "accuracy": accuracy}

  def create_graph_fn(self, input):
    is_training = (self.args.mode == "train")
    return self.net(input, self.args.num_classes,
                    is_training=is_training, data_format=self.args.data_format)

  def create_eval_metrics_fn(self, predictions, labels):
    equality = tf.equal(predictions["classes"],
                        tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy

  def create_loss_fn(self, logits, labels):
    loss_cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

    loss_l2 = self.l2_regularization()

    loss = tf.identity(loss_cross_entropy + loss_l2, "total_loss")

    return loss


def build(args):
  return ImageClassificationModeler(args)
