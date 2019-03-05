"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import six
from collections import Counter
import operator
import numpy as np
import re
import pickle

import tensorflow as tf

from .inputter import Inputter
from demo.text.preprocess import vocab_loader


def loadData(meta_data, unit):
  data = []
  if unit == "char":
    for meta in meta_data:
      with open(meta, 'rb') as f:
        d = f.read()
      if six.PY2:
        d = bytearray(d)
        data.extend([chr(c) for c in d])
  elif unit == "word":
    for meta in meta_data:
      with open(meta, 'rb') as f:
        d = f.read()
        d = re.findall(r"[\w']+|[:.,!?;\n]", d)
        data.extend(d)

  return data


class TextGenerationInputter(Inputter):
  def __init__(self, config, augmenter, encoder):
    super(TextGenerationInputter, self).__init__(config, augmenter)

    self.encoder = encoder

    if self.config.mode == "train":
      self.num_samples = 100000
      self.max_length = 50
    elif self.config.mode == "infer":
      self.num_samples = 256
      self.max_length = 1
    elif self.config.mode == "eval":
      self.num_samples = 10000
      self.max_length = 50
    elif self.config.mode == "export":
      self.num_samples = 1
      self.max_length = 1

    self.data = loadData(self.config.dataset_meta, self.config.unit)
    self.vocab, self.items, self.embd = vocab_loader.load(
      self.config.vocab_file, self.config.vocab_format, self.config.vocab_top_k)

    self.vocab_size = len(self.vocab)

    if self.config.mode == "train" or self.config.mode == "eval" or self.config.mode == "infer":
      # clean data
      if self.config.vocab_top_k > 0:
        self.data = [w for w in self.data if w in self.vocab]

      # encode data
      # Use the entire data here
      self.encode_data, self.encode_mask = self.encoder.encode([self.data], self.vocab, -1)
      self.encode_data = self.encode_data[0]
      self.encode_mask = self.encode_mask[0]

    if self.config.mode == "infer":
      self.config.starter = self.config.starter.split("#")
      self.config.starter = [[self.vocab[w] for w in self.config.starter if w in self.vocab]]

  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)
    max_step = (self.get_num_samples() * self.config.epochs // batch_size)
    tf.constant(max_step, name="max_step")

  def get_num_samples(self):
    return self.num_samples

  def get_vocab_size(self):
    return self.vocab_size

  def get_max_length(self):
    return self.max_length

  def get_items(self):
    return self.items

  def get_embd(self):
    return self.embd

  def get_starter(self):
    return self.config.starter

  def get_samples_fn(self):
    random_starts = np.random.randint(
      0,
      self.encode_data.shape[0] - self.max_length - 1,
      (self.num_samples,))

    for st in random_starts:
        seq = self.encode_data[st:st + self.max_length + 1]
        yield seq[:-1], seq[1:]

  def parse_fn(self, inputs, outputs):
    """Parse a single input sample
    """
    inputs.set_shape([self.max_length])
    outputs.set_shape([self.max_length])

    return (inputs, outputs)

  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count) 
    if self.config.mode == "export":
      input_item = tf.placeholder(tf.int32,
                             shape=(batch_size, self.max_length),
                             name="input_item")
      return input_item
    else:
      if self.config.mode == "train" or self.config.mode == "eval":

        dataset = tf.data.Dataset.from_generator(
          generator=lambda: self.get_samples_fn(),
          output_types=(tf.int32, tf.int32))

        dataset = dataset.repeat(self.config.epochs)

        dataset = dataset.map(
          lambda inputs, outputs: self.parse_fn(inputs, outputs),
          num_parallel_calls=4)

        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))

        dataset = dataset.prefetch(2)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
      else:
        return (tf.zeros([batch_size, self.max_length], tf.int32),
                tf.zeros([batch_size, self.max_length], tf.int32))


def build(config, augmenter, encoder):
  return TextGenerationInputter(config, augmenter, encoder)
