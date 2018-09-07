"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import time

import tensorflow as tf

from callback import Callback


class TrainSpeed(Callback):
  def __init__(self, args):
    super(TrainSpeed, self).__init__(args)
    self.graph = tf.get_default_graph()
    self.accumulated_num_samples = 0.0
    self.accumulated_time = 0.0
    self.batch_size = self.args.batch_size_per_gpu * self.args.num_gpu

  def before_run(self, sess, saver):
    pass

  def after_run(self, sess, saver):
    pass

  def before_step(self, sess):
    self.time_before_step = time.time()

  def after_step(self, sess, outputs_dict, saver):
    self.time_after_step = time.time()

    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    global_step = sess.run(global_step_op)

    self.accumulated_num_samples = self.accumulated_num_samples + self.batch_size
    self.accumulated_time = (self.accumulated_time + self.time_after_step -
                             self.time_before_step)

    every_n_iter = self.args.log_every_n_iter

    if global_step % every_n_iter == 0:
      num_samples_per_sec = self.accumulated_num_samples / self.accumulated_time
      print("speed: " + "{0:.4f}".format(num_samples_per_sec) + " samples/sec.")
      self.accumulated_num_samples = 0.0
      self.accumulated_time = 0.0


def build(args):
  return TrainSpeed(args)
