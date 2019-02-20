"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import os
import csv
import numpy as np

import tensorflow as tf

from .inputter import Inputter
from source.network.encoder import sentence

def loadSentences(dataset_meta):
  # Read sentences and labels from csv files
  sentences = []
  labels = []
  for meta in dataset_meta:
    dirname = os.path.dirname(meta)
    with open(meta) as f:
      parsed = csv.reader(f, delimiter="\t")
      for row in parsed:
        sentences.append(row[0].split(" "))
        labels.append([int(row[1])])
  return sentences, labels 


def loadVocab(vocab_file, top_k):
  # Read vocabulary
  # Every line has one word.
  # The embedding of the word is optoinally included in the same line

  vocab = []
  embd = []

  file = open(vocab_file,'r')
  count = 0
  for line in file.readlines():
      row = line.strip().split(' ')
      vocab.append(row[0])
      
      if len(row) > 1:
        embd.append(row[1:])

      count += 1
      if count == top_k:
        break

  file.close()

  vocab = { w : i for i, w in enumerate(vocab)}

  if embd:
    embd = np.asarray(embd).astype(np.float32)

  return vocab, embd


class TextClassificationInputter(Inputter):
  def __init__(self, config, augmenter, encoder):
    super(TextClassificationInputter, self).__init__(config, augmenter)

    self.encoder = encoder

    self.max_length = 256

    # Load data
    if self.config.mode == "train" or self.config.mode == "eval":
      for meta in self.config.dataset_meta:
        assert os.path.exists(meta), ("Cannot find dataset_meta file {}.".format(meta))
      self.sentences, self.labels = loadSentences(self.config.dataset_meta)
    elif self.config.mode == "infer":
      pass

    # Load vacabulary
    self.vocab, self.embd = loadVocab(self.config.vocab_file, self.config.vocab_top_k)

    # encode data
    self.encode_sentences, self.encode_masks = self.encoder.encode(self.sentences, self.vocab, self.max_length)

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
      pass
    else:
      if self.config.mode == "train" or self.config.mode == "eval" or self.config.mode == 'infer':

        dataset = tf.data.Dataset.from_generator(
          generator=lambda: self.get_samples_fn(),
          output_types=(tf.int32, tf.int32, tf.int32),
          output_shapes=(self.max_length, 1, self.max_length))

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
