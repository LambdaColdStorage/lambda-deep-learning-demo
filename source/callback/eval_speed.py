"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import time

import tensorflow as tf

from callback import Callback


class EvalSpeed(Callback):
  def __init__(self, args):
    super(EvalSpeed, self).__init__(args)

  def before_run(self, sess, saver):
    self.graph = tf.get_default_graph()
    self.accumulated_num_samples = 0.0
    self.accumulated_time = 0.0
    self.batch_size = self.args.batch_size_per_gpu * self.args.num_gpu
    self.global_step = 0.0

  def after_run(self, sess, saver, summary_writer):
    pass

  def before_step(self, sess):
    self.time_before_step = time.time()

  def after_step(self, sess, outputs_dict, saver, summary_writer):
    self.time_after_step = time.time()

    self.global_step = self.global_step + 1

    self.accumulated_num_samples = (self.accumulated_num_samples +
                                    self.batch_size)
    self.accumulated_time = (self.accumulated_time + self.time_after_step -
                             self.time_before_step)

    every_n_iter = self.args.log_every_n_iter

    if self.global_step % every_n_iter == 0:
      num_samples_per_sec = (self.accumulated_num_samples /
                             self.accumulated_time)
      self.accumulated_num_samples = 0.0
      self.accumulated_time = 0.0
      return {"speed": "Speed: " + "{0:.4f}".format(num_samples_per_sec)}
    else:
      return {}


def build(args):
  return EvalSpeed(args)
