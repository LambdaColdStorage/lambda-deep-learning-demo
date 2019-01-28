"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import sys
import time
import os

import matplotlib.pyplot as plt

import tensorflow as tf


class Runner(object):
  def __init__(self, config, inputter, modeler, callbacks):

    tf.reset_default_graph()

    self.config = config
    self.inputter = inputter
    self.modeler = modeler
    self.callbacks = callbacks

    self.modeler.get_dataset_info(self.inputter)

    self.session_config = self.create_session_config()
    self.sess = None

    self.feed_dict = {}

    self.outputs = None
    self.run_ops = []
    self.run_ops_names = []

  def create_session_config(self):
    """create session_config
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,
                                allow_growth=True)

    # set number of GPU devices
    device_count = {"GPU": self.config.gpu_count}

    session_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      device_count=device_count,
      gpu_options=gpu_options)

    return session_config

  def before_run(self):
    for callback in self.callbacks:
      callback.before_run(self.sess)

  def before_step(self):
    for callback in self.callbacks:
      callback.before_step(self.sess)

  def after_step(self):

    outputs_dict = {}
    for key, value in zip(self.run_ops_names, self.outputs):
      outputs_dict[key] = value

    print_msg = "\r"
    for callback in self.callbacks:
      return_dict = callback.after_step(self.sess, outputs_dict,
                                        self.feed_dict)
      if return_dict:
        for key in return_dict:
          print_msg = print_msg + return_dict[key] + " "

    if len(print_msg) > 0:
      print(print_msg, end='')
      sys.stdout.flush()

  def after_run(self):
    for callback in self.callbacks:
      callback.after_run(self.sess)

  def prepare_feed_dict(self):

      # Get the pre-computation feed_dict
      for key in self.modeler.feed_dict_pre:
        if isinstance(self.modeler.feed_dict_pre[key], tf.Tensor):
          self.feed_dict[key] = self.sess.run(
            self.modeler.feed_dict_pre[key])
        else:
          self.feed_dict[key] = self.modeler.feed_dict_pre[key]

      # Get the sequential feed_dict (Updated by previous step's output)
      for key in self.modeler.feed_dict_seq:
        if isinstance(self.modeler.feed_dict_seq[key], tf.Tensor):
          self.feed_dict[key] = self.sess.run(
            self.modeler.feed_dict_seq[key])
        else:
          self.feed_dict[key] = self.modeler.feed_dict_seq[key]

  def collect_summary(self, run_ops_names, run_ops):
    for name, op in zip(run_ops_names, run_ops):
      if name in self.config.summary_names:
        tf.summary.scalar(name, op)
    return tf.summary.merge_all()

  def collect_ops(self, ops):
    # Create train_op for gradient, keep other ops unchanged
    run_ops = []
    run_ops_names = []

    for key in ops:
      if key == "grads":
        minimize_op = self.modeler.optimizer.apply_gradients(
          ops[key], global_step=self.modeler.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        op = tf.group(minimize_op, update_ops)
      else:
        op = ops[key]
      run_ops.append(op)
      run_ops_names.append(key)

    if self.config.mode == "train":
      summary_op = self.collect_summary(run_ops_names, run_ops)
      run_ops.append(summary_op)
      run_ops_names.append("summary")

    return run_ops, run_ops_names

  def print_trainable_variables(self):

    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
      print (i)

    print(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

  def print_global_variables(self):

    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      print(i.name)

    print(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

  def run(self):

    if self.config.mode == "export":
      # inputs = self.inputter.input_fn()

      outputs = self.modeler.model_fn(self.inputter.input_fn())

      for op in tf.get_default_graph().get_operations():
          print(str(op.name))

      with tf.Session(config=self.session_config) as self.sess:
        self.before_run()
    else:
      self.create_graph()

      # self.print_global_variables()

      with tf.Session(config=self.session_config) as self.sess:

        # Before run
        self.before_run()

        self.prepare_feed_dict()

        global_step = 0
        if self.config.mode == "train":
          global_step = self.sess.run(self.global_step_op)

        max_step = self.sess.run(self.max_step_op)

        while global_step < max_step:
          self.before_step()

          self.outputs = self.sess.run(self.run_ops,
                                       feed_dict=self.feed_dict)
          self.after_step()

          global_step = global_step + 1

        self.after_run()

  def dev2(self):
    self.create_graph()

    self.print_trainable_variables()


  def dev(self):
    nonreplicated_fns = [self.modeler.create_nonreplicated_fn,
                         self.inputter.create_nonreplicated_fn]

    for fn in nonreplicated_fns:
      fn()

    # image_id, image, gt_labels, gt_bboxes, gt_mask, scale, translation, file_name
    batch = self.inputter.input_fn()

    results = self.modeler.model_fn(batch)

    # self.print_trainable_variables()

    with tf.Session(config=self.session_config) as self.sess:
      self.sess.run(tf.global_variables_initializer())

      num_batch = 1

      # Test input_fn
      for i in range(num_batch):
        # _batch = self.sess.run(batch)
        _results = self.sess.run(results)
        print(_results[0].shape)
        print(_results[1].shape)
        print(_results[2].shape)
        print(_results[3].shape)
        print(_results[4].shape)
        print(_results[5].shape)
        print(_results[6].shape)


def build(config, inputter, modeler, callbacks):
  return Runner(config, inputter, modeler, callbacks)
