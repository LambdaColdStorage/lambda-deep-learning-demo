"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function

import os
import sys
import glob

import tensorflow as tf

from .callback import Callback


class TrainBasic(Callback):
  def __init__(self, config):
    super(TrainBasic, self).__init__(config)

  def before_run(self, sess):
    self.graph = tf.get_default_graph()

    # Create saver
    self.saver = tf.train.Saver(
      max_to_keep=self.config.keep_checkpoint_max,
      name="global_saver")

    if not os.path.isdir(self.config.model_dir):
      os.makedirs(self.config.model_dir)

    if tf.train.checkpoint_exists(
      os.path.join(self.config.model_dir, "*ckpt*")):
      self.saver.restore(sess,
                         tf.train.latest_checkpoint(
                           self.config.model_dir))
      print("Parameters restored.")
    else:
      print("Initialize global variables ... ")
      sess.run(tf.global_variables_initializer())

    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    max_step_op = self.graph.get_tensor_by_name("max_step:0")
    global_step = sess.run(global_step_op)
    max_step = sess.run(max_step_op)

    if global_step >= max_step:
      sys.exit("Training has already reached the maximum steps.")
    else:
      if global_step == 0:
        print("Start training from step " + str(global_step))

        # Restore some weights from pre-trained model
        if self.config.pretrained_model:
          self.config.pretrained_model = os.path.expanduser(
            self.config.pretrained_model)
          print("Try to initialize weights from pre-trained model.")
          if tf.train.checkpoint_exists(self.config.pretrained_model):
            # Need this for batchnorm
            # variables_to_restore = {v.name.split(":")[0]: v
            #                         for v in tf.get_collection(
            #                             tf.GraphKeys.GLOBAL_VARIABLES)}

            # This will work for restoring weights from a model trained with different optimizer
            variables_to_restore = {v.name.split(":")[0]: v
                                    for v in tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES)}

            if self.config.skip_pretrained_var:
              variables_to_restore = {
                v: variables_to_restore[v] for
                v in variables_to_restore if not
                any(x in v for
                    x in self.config.skip_pretrained_var)}

            if variables_to_restore:
              saver_pre_trained = tf.train.Saver(
                var_list=variables_to_restore)
              saver_pre_trained.restore(sess, self.config.pretrained_model)

              print("Weights restored from pre-trained model.")
            else:
              print("Found no useful weights")
          else:
            print("Can't find pre-trained model at " +
                  self.config.pretrained_model)
            print("Initialize weight randomly.")
      else:
        print("Resume training from step " + str(global_step))

  def after_run(self, sess):
    max_step_op = self.graph.get_tensor_by_name("max_step:0")
    max_step = sess.run(max_step_op)

    if max_step % self.config.save_checkpoints_steps != 0:
      print("\nSaving checkpoint for the final step ...")
      save_path = self.saver.save(sess,
                                  os.path.join(self.config.model_dir,
                                               "model.ckpt"),
                                  global_step=max_step)
      print("Checkpoint " + save_path + " has been saved.")

  def after_step(self, sess, outputs_dict, feed_dict=None):


    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    global_step = sess.run(global_step_op)

    if global_step % self.config.save_checkpoints_steps == 0:
      save_path = self.saver.save(
        sess,
        os.path.join(self.config.model_dir,
                     "model.ckpt"),
        global_step=global_step)
      print("Saving checkpoint " + save_path)


def build(config):
  return TrainBasic(config)
