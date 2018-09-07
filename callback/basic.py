"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import os
import sys
import time

import tensorflow as tf

from callback import Callback


class Basic(Callback):
  def __init__(self, args):
    super(Basic, self).__init__(args)
    self.graph = tf.get_default_graph()
    self.accumulated_loss = 0.0
    self.accumulated_samples = 0.0
    self.accumulated_time = 0.0
    self.batch_size = self.args.batch_size_per_gpu * self.args.num_gpu

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
      if self.args.mode == "train":
        print("Initialize global variables ... ")
        sess.run(tf.global_variables_initializer())
      else:
        assert False, "Can not find checkpiont for {}".format(self.args.mode)

    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    max_step_op = self.graph.get_tensor_by_name("max_step:0")
    global_step = sess.run(global_step_op)
    max_step = sess.run(max_step_op)

    if self.args.mode == "train":
      if global_step >= max_step:
        # print("Training has already reached the maximum steps.")
        sys.exit("Training has already reached the maximum steps.")
      else:
        if global_step == 0:
          print("Start training from step " + str(global_step))
        else:
          print("Resume training from step " + str(global_step))
    elif self.args.mode == "eval":
      pass
    elif self.args.mode == "infer":
      pass
    else:
      pass

  def after_run(self, sess, saver):
    if self.args.mode == "train":
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
    self.time_before_step = time.time()

  def after_step(self, sess, outputs_dict, saver):
    self.time_after_step = time.time()

    global_step_op = self.graph.get_tensor_by_name("global_step:0")

    global_step = sess.run(global_step_op)

    self.accumulated_samples = self.accumulated_samples + self.batch_size
    self.accumulated_time = (self.accumulated_time + self.time_after_step -
                             self.time_before_step)
    self.accumulated_loss = self.accumulated_loss + outputs_dict["loss"]

    if self.args.mode == "train":
      every_n_iter = self.args.log_every_n_iter

      if global_step % every_n_iter == 0:
        num_samples_per_sec = self.accumulated_samples / self.accumulated_time
        running_loss = self.accumulated_loss / every_n_iter
        print("step: " + str(global_step) +
              ", loss: " + "{0:.4f}".format(running_loss) +
              ", speed: " + "{0:.4f}".format(num_samples_per_sec) +
              " samples/sec.")
        self.accumulated_samples = 0.0
        self.accumulated_time = 0.0
        self.accumulated_loss = 0.0

      if global_step % self.args.save_checkpoints_steps == 0:
        save_path = saver.save(
          sess,
          os.path.join(self.args.model_dir,
                       "model.ckpt"),
          global_step=global_step)
        print("Saving checkpoint " + save_path)


def build(args):
  return Basic(args)
