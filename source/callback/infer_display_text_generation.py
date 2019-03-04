"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function

import numpy as np
import math

import tensorflow as tf

from .callback import Callback


def pick(prob):
    t = np.cumsum(prob)
    s = np.sum(prob)
    return(int(math.floor(np.searchsorted(t, 0.9999 * np.random.rand(1) * s))))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class InferDisplayTextGeneration(Callback):
  def __init__(self, config):
    super(InferDisplayTextGeneration, self).__init__(config)
    self.input = ""
    self.output = ""

  def before_run(self, sess):
    self.graph = tf.get_default_graph()

  def after_run(self, sess):
    print('-------------------------------------------------')
    if self.config.unit == "char":
      print(self.input[0] + self.output)
    elif self.config.unit == "word":
      print(self.input.split()[0] + " " + self.output)
    print('-------------------------------------------------')

  def after_step(self, sess, outputs_dict, feed_dict=None):
    items = outputs_dict["items"]
    for i, l in zip(outputs_dict["inputs"], outputs_dict["logits"]):

      if self.config.unit == "char":
        self.input += items[i[0]]
      elif self.config.unit == "word":
        self.input += " " + items[i[0]]

      # self.input += items[i[0]]
      p = softmax(l / self.config.softmax_temperature)
      pick_id = pick(p)

      if self.config.unit == "char":
        self.output += items[pick_id]
      elif self.config.unit == "word":
        if items[pick_id] == "\n":
          self.output += items[pick_id]
        else:
          self.output += items[pick_id] + " "

      # Get the placeholder for inputs and states
      inputs_place_holder = self.graph.get_tensor_by_name("RNN/inputs:0")
      c0_place_holder = self.graph.get_tensor_by_name("RNN/c0:0")
      h0_place_holder = self.graph.get_tensor_by_name("RNN/h0:0")
      c1_place_holder = self.graph.get_tensor_by_name("RNN/c1:0")
      h1_place_holder = self.graph.get_tensor_by_name("RNN/h1:0")

      # Python passes dictionary by reference
      feed_dict[inputs_place_holder] = np.array([[pick_id]], dtype=np.int32)
      feed_dict[c0_place_holder] = outputs_dict["last_state"][0][0]
      feed_dict[h0_place_holder] = outputs_dict["last_state"][0][1]
      feed_dict[c1_place_holder] = outputs_dict["last_state"][1][0]
      feed_dict[h1_place_holder] = outputs_dict["last_state"][1][1]


def build(config):
  return InferDisplayTextGeneration(config)
