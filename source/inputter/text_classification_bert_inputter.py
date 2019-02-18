"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import os
import csv
import re

import six
from collections import Counter
import operator
import numpy as np
import pickle

import tensorflow as tf

from .inputter import Inputter


class TextClassificationBertInputter(Inputter):
  def __init__(self, config, augmenter):
    super(TextClassificationBertInputter, self).__init__(config, augmenter)

    self.max_seq_length = 256

    if self.config.mode == "train":
      self.num_samples = 25000
    elif self.config.mode == "eval":
      self.num_samples = 25000
    else:
      self.num_samples = 1

    self.num_classes = 2

    self.name_to_features = {
        "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }


  def parse_fn(self, record):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, self.name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return (example["input_ids"], example["input_mask"],
            example["segment_ids"], example["label_ids"], example["is_real_example"])

  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)
    max_step = (self.get_num_samples() * self.config.epochs // batch_size)
    tf.constant(max_step, name="max_step")

  def get_num_samples(self):
    return self.num_samples

  def get_num_classes(self):
    return self.num_classes

  def get_num_epochs(self):
    return self.config.epochs

  def get_vocab_size(self):
    pass
    # return self.vocab_size

  def get_words(self):
    pass
    # return self.words

  def get_seq_length(self):
    pass
    # return self.seq_length

  def get_samples_fn(self):
    pass
    # for sentence, label in zip(self.encode_sentences, self.labels):
    #   yield sentence, label 

  def input_fn(self, test_samples=[]):
    
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count) 
    if self.config.mode == "export":
      pass
    else:
      if self.config.mode == "train" or self.config.mode == "eval" or self.config.mode == 'infer':

        dataset = tf.data.TFRecordDataset(self.config.dataset_meta)

        if self.config.mode == "train":
          dataset = dataset.shuffle(self.get_num_samples())

        dataset = dataset.repeat(self.config.epochs)

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda record: self.parse_fn(record),
                batch_size=batch_size,
                drop_remainder=True))

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def build(config, augmenter):
  return TextClassificationBertInputter(config, augmenter)
