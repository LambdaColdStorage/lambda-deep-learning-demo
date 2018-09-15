"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from callback import Callback


class TrainAccuracy(Callback):
  def __init__(self, config):
    super(TrainAccuracy, self).__init__(config)

  def before_run(self, sess):
    self.graph = tf.get_default_graph()
    self.accumulated_accuracy = 0.0

  def after_step(self, sess, outputs_dict, feed_dict=None):

    global_step_op = self.graph.get_tensor_by_name("global_step:0")

    global_step = sess.run(global_step_op)

    self.accumulated_accuracy = (self.accumulated_accuracy +
                                 outputs_dict["accuracy"])

    every_n_iter = self.config.log_every_n_iter

    if global_step % every_n_iter == 0:
      running_accuracy = self.accumulated_accuracy / every_n_iter
      self.accumulated_accuracy = 0.0
      return {"accuracy": "Accuracy: " + "{0:.4f}".format(running_accuracy)}
    else:
      return {}


def build(config):
  return TrainAccuracy(config)
