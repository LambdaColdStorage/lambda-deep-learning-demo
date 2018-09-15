"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from callback import Callback


class EvalLoss(Callback):
  def __init__(self, config):
    super(EvalLoss, self).__init__(config)

  def before_run(self, sess):
    self.graph = tf.get_default_graph()
    self.accumulated_loss = 0.0
    self.global_step = 0.0

  def after_run(self, sess):
    eval_loss = self.accumulated_loss / self.global_step
    print("Evaluation loss: " + "{0:.4f}".format(eval_loss))

  def after_step(self, sess, outputs_dict, feed_dict=None):
    self.global_step = self.global_step + 1

    self.accumulated_loss = (self.accumulated_loss +
                             outputs_dict["loss"])

    every_n_iter = self.config.log_every_n_iter

    if self.global_step % every_n_iter == 0:
      running_loss = self.accumulated_loss / self.global_step
      return {"loss": "Loss: " + "{0:.4f}".format(running_loss)}
    else:
      return {}


def build(config):
  return EvalLoss(config)
