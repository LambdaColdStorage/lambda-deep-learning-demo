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

from callback import Callback


class TrainBasic(Callback):
  def __init__(self, args):
    super(TrainBasic, self).__init__(args)
    self.graph = tf.get_default_graph()

  def before_run(self, sess, saver):
    print("Basic callback before_run")

    if not os.path.isdir(self.args.model_dir):
      os.makedirs(self.args.model_dir)

    if tf.train.checkpoint_exists(
      os.path.join(self.args.model_dir, "*ckpt*")):
      saver.restore(sess,
                    tf.train.latest_checkpoint(
                      self.args.model_dir))
      print("Parameters restored.")
    else:
      print("Initialize global variables ... ")
      sess.run(tf.global_variables_initializer())

    # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #   print (i.name)

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
        if self.args.pretrained_dir:
          self.args.pretrained_dir = os.path.expanduser(
            self.args.pretrained_dir)
          print("Try to initialize weights from pre-trained model.")
          if tf.train.checkpoint_exists(
            os.path.join(self.args.pretrained_dir, "*ckpt*")):
            variables_to_restore = {v.name.split(":")[0]: v
                                    for v in tf.get_collection(
                                        tf.GraphKeys.GLOBAL_VARIABLES)}
            if self.args.skip_pretrained_var_list:
              variables_to_restore = {
                v: variables_to_restore[v] for
                v in variables_to_restore if not
                any(x in v for
                    x in self.args.skip_pretrained_var_list)}

            if variables_to_restore:
              saver_pre_trained = tf.train.Saver(
                var_list=variables_to_restore)
              for file in glob.glob(self.args.pretrained_dir + "/*.ckpt"):
                ckpt_file = file
              saver_pre_trained.restore(sess,
                  os.path.join(self.args.pretrained_dir, ckpt_file))

              print("Weights restored from pre-trained model.")
            else:
              print("Found no useful weights")
          else:
            print("Can't find pre-trained model at " + self.args.pretrained_dir)
            print("Initialize weight randomly.")
      else:
        print("Resume training from step " + str(global_step))

  def after_run(self, sess, saver, summary_writer):
    max_step_op = self.graph.get_tensor_by_name("max_step:0")
    max_step = sess.run(max_step_op)

    if max_step % self.args.save_checkpoints_steps != 0:
      print("\nSaving checkpoint for the final step ...")
      save_path = saver.save(sess,
                             os.path.join(self.args.model_dir,
                                          "model.ckpt"),
                             global_step=max_step)
      print("Checkpoint " + save_path + " has been saved.")

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver, summary_writer):

    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    global_step = sess.run(global_step_op)

    if global_step % self.args.save_checkpoints_steps == 0:
      save_path = saver.save(
        sess,
        os.path.join(self.args.model_dir,
                     "model.ckpt"),
        global_step=global_step)
      print("Saving checkpoint " + save_path)


def build(args):
  return TrainBasic(args)
