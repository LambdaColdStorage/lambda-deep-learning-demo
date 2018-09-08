"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import sys

import tensorflow as tf


class Runner(object):
  def __init__(self, args, inputter, modeler):
    self.args = args
    self.inputter = inputter
    self.modeler = modeler

    self.modeler.num_samples = self.inputter.get_num_samples()

    self.session_config = self.create_session_config()
    self.sess = None
    self.batch_size = self.args.batch_size_per_gpu * self.args.num_gpu
    self.feed_dict = {}
    self.outputs = None
    self.nonreplicated_fns = [self.modeler.create_nonreplicated_fn,
                              self.inputter.create_nonreplicated_fn]
    self.run_ops = []
    self.run_ops_names = []

  def before_run(self, callbacks):
    for callback in callbacks:
      callback.before_run(self.sess, self.saver)

    self.run_feed_dict()

  def before_step(self, callbacks):
    for callback in callbacks:
      callback.before_step(self.sess)

  def after_step(self, callbacks):

    outputs_dict = {}
    for key, value in zip(self.run_ops_names, self.outputs):
      outputs_dict[key] = value

    print_msg = "\r"
    for callback in callbacks:
      return_dict = callback.after_step(self.sess, outputs_dict, self.saver)
      if return_dict:
        for key in return_dict:
          print_msg = print_msg + return_dict[key] + " "

    if len(print_msg) > 0:
      print(print_msg, end='')
      sys.stdout.flush()

  def after_run(self, callbacks):
    for callback in callbacks:
      callback.after_run(self.sess, self.saver)

  def run_feed_dict(self):
      for key in self.modeler.feed_dict_ops:
        self.feed_dict[key] = self.sess.run(
          self.modeler.feed_dict_ops[key])

  def run(self):
    self.create_graph()

    with tf.Session(config=self.session_config) as self.sess:

      # Before run
      self.before_run(self.modeler.callbacks)

      global_step = 0
      if self.args.mode == "train":
        global_step = self.sess.run(self.global_step_op)

      max_step = self.sess.run(self.max_step_op)

      while global_step < max_step:
        self.before_step(self.modeler.callbacks)

        self.outputs = self.sess.run(self.run_ops, feed_dict=self.feed_dict)

        self.after_step(self.modeler.callbacks)

        global_step = global_step + 1

      self.after_run(self.modeler.callbacks)


def build(args, inputter, modeler):
  return Runner(args, inputter, modeler)
