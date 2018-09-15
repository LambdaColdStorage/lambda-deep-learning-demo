"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import os
import sys

import tensorflow as tf

from callback import Callback


class EvalBasic(Callback):
  def __init__(self, config):
    super(EvalBasic, self).__init__(config)

  def before_run(self, sess, saver):
    self.graph = tf.get_default_graph()
    ckpt_path = os.path.join(self.config.model_dir, "*ckpt*")
    if tf.train.checkpoint_exists(ckpt_path):
      saver.restore(sess,
                    tf.train.latest_checkpoint(self.config.model_dir))
      print("Parameters restored.")
    else:
      sys.exit("Can not find checkpoint at " + ckpt_path)

    print("Start evaluation.")

  def after_run(self, sess, saver, summary_writer):
    print("\n")


def build(config):
  return EvalBasic(config)
