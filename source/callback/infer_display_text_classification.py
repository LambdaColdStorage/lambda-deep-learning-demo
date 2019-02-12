"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from .callback import Callback


class InferDisplayTextClassification(Callback):
  def __init__(self, config):
    super(InferDisplayTextClassification, self).__init__(config)

  def before_run(self, sess):
    self.graph = tf.get_default_graph()

  def after_step(self, sess, outputs_dict, feed_dict=None):
    for p, c in zip(outputs_dict["probabilities"],
                    outputs_dict["classes"]):
      print("Predict: " + str(c) + ", Probability: " + str(p[c]))


def build(config):
  return InferDisplayTextClassification(config)
