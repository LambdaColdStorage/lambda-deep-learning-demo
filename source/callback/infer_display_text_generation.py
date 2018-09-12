"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function

import numpy as np

import tensorflow as tf

from callback import Callback


def pick(prob):
    t = np.cumsum(prob)
    s = np.sum(prob)
    return(int(np.searchsorted(t, np.random.rand(1) * s)))


class InferDisplayTextGeneration(Callback):
  def __init__(self, args):
    super(InferDisplayTextGeneration, self).__init__(args)
    self.output = ""

  def before_run(self, sess, saver):
    self.graph = tf.get_default_graph()

  def after_run(self, sess, saver, summary_writer):
    print(self.output)

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver, summary_writer):
    chars = outputs_dict["chars"]
    for p in outputs_dict["probabilities"]:
      # print(chars[pick(p)])
      self.output += chars[pick(p)]


def build(args):
  return InferDisplayTextGeneration(args)
