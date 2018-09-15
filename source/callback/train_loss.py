"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from callback import Callback


class TrainLoss(Callback):
  def __init__(self, config):
    super(TrainLoss, self).__init__(config)

  def before_run(self, sess):
    self.graph = tf.get_default_graph()
    self.accumulated_loss = 0.0

  def after_step(self, sess, outputs_dict, feed_dict=None):
    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    global_step = sess.run(global_step_op)

    self.accumulated_loss = self.accumulated_loss + outputs_dict["loss"]

    every_n_iter = self.config.log_every_n_iter

    if global_step % every_n_iter == 0:
      loss = self.accumulated_loss / every_n_iter
      self.accumulated_loss = 0.0
      # print("loss: " + "{0:.4f}".format(loss))
      return {"loss": "Loss: " + "{0:.4f}".format(loss)}
    else:
      return {}


def build(config):
  return TrainLoss(config)
