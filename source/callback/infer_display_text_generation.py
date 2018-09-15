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
  def __init__(self, config):
    super(InferDisplayTextGeneration, self).__init__(config)
    self.input = ""
    self.output = ""

  def before_run(self, sess):
    self.graph = tf.get_default_graph()

  def after_run(self, sess):
    print('-------------------------------------------------')
    print(self.input[0] + self.output)
    print('-------------------------------------------------')

  def after_step(self, sess, outputs_dict, feed_dict=None):
    chars = outputs_dict["chars"]
    for i, p in zip(outputs_dict["inputs"], outputs_dict["probabilities"]):

      self.input += chars[i[0]]

      pick_id = pick(p)

      self.output += chars[pick_id]

      # Get the placeholder for inputs and states
      inputs_place_holder = self.graph.get_tensor_by_name("CharRNN/inputs:0")
      c0_place_holder = self.graph.get_tensor_by_name("CharRNN/c0:0")
      h0_place_holder = self.graph.get_tensor_by_name("CharRNN/h0:0")
      c1_place_holder = self.graph.get_tensor_by_name("CharRNN/c1:0")
      h1_place_holder = self.graph.get_tensor_by_name("CharRNN/h1:0")

      # Python passes dictionary by reference
      feed_dict[inputs_place_holder] = np.array([[pick_id]], dtype=np.int32)
      feed_dict[c0_place_holder] = outputs_dict["last_state"][0][0]
      feed_dict[h0_place_holder] = outputs_dict["last_state"][0][1]
      feed_dict[c1_place_holder] = outputs_dict["last_state"][1][0]
      feed_dict[h1_place_holder] = outputs_dict["last_state"][1][1]


def build(config):
  return InferDisplayTextGeneration(config)
