"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import os
import sys

import tensorflow as tf

from callback import Callback


class InferBasic(Callback):
  def __init__(self, args):
    super(InferBasic, self).__init__(args)

  def before_run(self, sess, saver):
    self.graph = tf.get_default_graph()
    ckpt_path = os.path.join(self.args.model_dir, "*ckpt*")
    if tf.train.checkpoint_exists(ckpt_path):
      saver.restore(sess,
                    tf.train.latest_checkpoint(self.args.model_dir))
      print("Parameters restored.")
    else:
      sys.exit("Can not find checkpoint at " + ckpt_path)

    print("Start inference.")


def build(args):
  return InferBasic(args)
