"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from callback import Callback


class InferDisplayImageClassification(Callback):
  def __init__(self, args):
    super(InferDisplayImageClassification, self).__init__(args)
    self.graph = tf.get_default_graph()

  def before_run(self, sess, saver):
    pass

  def after_run(self, sess, saver, summary_writer):
    pass

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver, summary_writer):
    for p, c in zip(outputs_dict["probabilities"],
                    outputs_dict["classes"]):
      print("Predict: " + str(c) + ", Probability: " + str(p[c]))


def build(args):
  return InferDisplayImageClassification(args)
