"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import os
import csv
import numpy as np
import re
import pickle

import tensorflow as tf

from .inputter import Inputter
from demo.text.preprocess import vocab_loader


def loadData(data, mode):
  sentences = []
  labels = []

  if mode == "train" or mode == "eval":
    # Training and evaluation use clean data
    for meta in data:
      dirname = os.path.dirname(meta)
      with open(meta) as f:
        parsed = csv.reader(f, delimiter="\t")
        for row in parsed:
          sentences.append(row[0].split(" "))
          labels.append([int(row[1])])
  elif mode == "infer":
    # Inference use sentences that are not cleaned
    for s in data:
      sentences.append(re.findall(r"[\w']+|[.,!?;]", s))
      labels.append([int(-1)])

  return sentences, labels 


class TextClassificationInputter(Inputter):
  def __init__(self, config, augmenter, encoder):
    super(TextClassificationInputter, self).__init__(config, augmenter)

    self.encoder = encoder

    # Load vocabulary
    self.vocab, self.items, self.embd = vocab_loader.load(
      self.config.vocab_file, self.config.vocab_format, self.config.vocab_top_k)

    # Load data
    if self.config.mode == "export":
      self.num_samples = 1
    else:
      if self.config.mode == "train" or self.config.mode == "eval":
        for meta in self.config.dataset_meta:
          assert os.path.exists(meta), ("Cannot find dataset_meta file {}.".format(meta))
        self.sentences, self.labels = loadData(self.config.dataset_meta, self.config.mode)
      elif self.config.mode == "infer":
        self.sentences, self.labels = loadData(self.config.test_samples, self.config.mode)

      # encode data
      self.encode_sentences, self.encode_masks = self.encoder.encode(self.sentences, self.vocab, self.config.max_length)

      self.num_samples = len(self.encode_sentences)

  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)
    max_step = (self.get_num_samples() * self.config.epochs // batch_size)
    tf.constant(max_step, name="max_step")

  def get_num_samples(self):
    return self.num_samples

  def get_vocab_size(self):
    return len(self.vocab)

  def get_embd(self):
    return self.embd

  def get_num_epochs(self):
    return self.config.epochs

  def get_samples_fn(self):
    for encode_sentence, label, mask in zip(self.encode_sentences, self.labels, self.encode_masks):
      yield encode_sentence, label, mask

  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count) 
    if self.config.mode == "export":
      encode_sentence = tf.placeholder(tf.int32,
                             shape=(batch_size, self.config.max_length),
                             name="input_text")
      mask = tf.placeholder(tf.int32,
                             shape=(batch_size, self.config.max_length),
                             name="input_mask")
      return encode_sentence, mask
    else:
      if self.config.mode == "train" or self.config.mode == "eval" or self.config.mode == 'infer':

        dataset = tf.data.Dataset.from_generator(
          generator=lambda: self.get_samples_fn(),
          output_types=(tf.int32, tf.int32, tf.int32),
          output_shapes=(self.config.max_length, 1, self.config.max_length))

        if self.config.mode == "train":
          dataset = dataset.shuffle(self.get_num_samples())

        dataset = dataset.repeat(self.config.epochs)

        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))

        dataset = dataset.prefetch(2)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def build(config, augmenter, encoder):
  return TextClassificationInputter(config, augmenter, encoder)
