"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from callback import Callback


class RunningAccuracy(Callback):
  def __init__(self, args):
    super(RunningAccuracy, self).__init__(args)
    self.graph = tf.get_default_graph()
    self.accumulated_accuracy = 0.0
    self.global_step = 0.0

  def before_run(self, sess, saver):
    pass

  def after_run(self, sess, saver):
    eval_accuracy = self.accumulated_accuracy / self.global_step
    print("Evaluation accuracy: " + "{0:.4f}".format(eval_accuracy))

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver):

    self.global_step = self.global_step + 1

    self.accumulated_accuracy = (self.accumulated_accuracy +
                                 outputs_dict["accuracy"])

    every_n_iter = self.args.log_every_n_iter

    if self.global_step % every_n_iter == 0:
      running_accuracy = self.accumulated_accuracy / self.global_step
      print("accuracy: " + "{0:.4f}".format(running_accuracy))


def build(args):
  return RunningAccuracy(args)
