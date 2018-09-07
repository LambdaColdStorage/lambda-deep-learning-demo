"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import os
import sys

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

    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    max_step_op = self.graph.get_tensor_by_name("max_step:0")
    global_step = sess.run(global_step_op)
    max_step = sess.run(max_step_op)

    if global_step >= max_step:
      # print("Training has already reached the maximum steps.")
      sys.exit("Training has already reached the maximum steps.")
    else:
      if global_step == 0:
        print("Start training from step " + str(global_step))
      else:
        print("Resume training from step " + str(global_step))

  def after_run(self, sess, saver):
    max_step_op = self.graph.get_tensor_by_name("max_step:0")
    max_step = sess.run(max_step_op)

    if max_step % self.args.save_checkpoints_steps != 0:
      print("Saving checkpoint for the final step ...")
      save_path = saver.save(sess,
                             os.path.join(self.args.model_dir,
                                          "model.ckpt"),
                             global_step=max_step)
      print("Checkpoint " + save_path + " has been saved.")

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver):

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
