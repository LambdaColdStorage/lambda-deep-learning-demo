"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from callback import Callback


class EvalAccuracy(Callback):
  def __init__(self, args):
    super(EvalAccuracy, self).__init__(args)

  def before_run(self, sess, saver):
    self.graph = tf.get_default_graph()
    self.accumulated_accuracy = 0.0
    self.global_step = 0.0

  def after_run(self, sess, saver, summary_writer):
    eval_accuracy = self.accumulated_accuracy / self.global_step
    print("Evaluation accuracy: " + "{0:.4f}".format(eval_accuracy))

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver, summary_writer, feed_dict=None):

    self.global_step = self.global_step + 1

    self.accumulated_accuracy = (self.accumulated_accuracy +
                                 outputs_dict["accuracy"])

    every_n_iter = self.args.log_every_n_iter

    if self.global_step % every_n_iter == 0:
      running_accuracy = self.accumulated_accuracy / self.global_step
      return {"accuracy": "Accuracy: " + "{0:.4f}".format(running_accuracy)}
    else:
      return {}


def build(args):
  return EvalAccuracy(args)
