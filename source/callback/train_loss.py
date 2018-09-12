"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from callback import Callback


class TrainLoss(Callback):
  def __init__(self, args):
    super(TrainLoss, self).__init__(args)

  def before_run(self, sess, saver):
    self.graph = tf.get_default_graph()
    self.accumulated_loss = 0.0

  def after_run(self, sess, saver, summary_writer):
    pass

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver, summary_writer, feed_dict=None):
    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    global_step = sess.run(global_step_op)

    self.accumulated_loss = self.accumulated_loss + outputs_dict["loss"]

    every_n_iter = self.args.log_every_n_iter

    if global_step % every_n_iter == 0:
      loss = self.accumulated_loss / every_n_iter
      self.accumulated_loss = 0.0
      # print("loss: " + "{0:.4f}".format(loss))
      return {"loss": "Loss: " + "{0:.4f}".format(loss)}
    else:
      return {}


def build(args):
  return TrainLoss(args)
