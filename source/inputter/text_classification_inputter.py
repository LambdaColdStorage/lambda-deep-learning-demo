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


RNN_SIZE = 256


def loadSentences(dataset_meta):
  sentences = []
  labels = []
  for meta in dataset_meta:
    dirname = os.path.dirname(meta)
    with open(meta) as f:
      parsed = csv.reader(f, delimiter="\t")
      for row in parsed:
        
        # sentences.append(row[3].split())
        sentences.append(re.findall(r"[\w']+|[.,!?;]", row[3]))
        
        labels.append([int(row[1])])
  return sentences, labels    

def loadSentences_from_list(list_sentences):
  sentences = []
  labels = []
  for s in list_sentences:
    sentences.append(re.findall(r"[\w']+|[.,!?;]", s))
    labels.append([int(-1)])
  return sentences, labels     

def encodeSetences(sentences, words2idx):
  encode_sentences = [np.array([words2idx[w] for w in s if w in words2idx], dtype='int32') for s in sentences]
  return encode_sentences


class TextClassificationInputter(Inputter):
  def __init__(self, config, augmenter):
    super(TextClassificationInputter, self).__init__(config, augmenter)

    f = open(self.config.vocab_file, 'rb')
    self.words2idx, self.words = pickle.load(f)
    f.close()

    if self.config.mode == 'train' or self.config.mode == 'eval':
      for meta in self.config.dataset_meta:
        assert os.path.exists(meta), ("Cannot find dataset_meta file {}.".format(meta))      
      self.sentences, self.labels = loadSentences(self.config.dataset_meta)
    else:
      self.sentences, self.labels = loadSentences_from_list(self.config.test_samples)

    self.encode_sentences = encodeSetences(self.sentences, self.words2idx)

    self.num_samples = len(self.encode_sentences)
    self.vocab_size = len(self.words2idx)

  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)
    max_step = (self.get_num_samples() * self.config.epochs // batch_size)
    tf.constant(max_step, name="max_step")

  def get_num_samples(self):
    return self.num_samples

  def get_vocab_size(self):
    return self.vocab_size

  def get_words(self):
    return self.words

  def get_seq_length(self):
    return self.seq_length

  def get_samples_fn(self):
    for sentence, label in zip(self.encode_sentences, self.labels):
      yield sentence, label 

  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count) 
    if self.config.mode == "export":
      pass
    #   input_chars = tf.placeholder(tf.int32,
    #                          shape=(batch_size, self.seq_length),
    #                          name="input_chars")
    #   c0 = tf.placeholder(
    #     tf.float32,
    #     shape=(batch_size, RNN_SIZE), name="c0")
    #   h0 = tf.placeholder(
    #     tf.float32,
    #     shape=(batch_size, RNN_SIZE), name="h0")
    #   c1 = tf.placeholder(
    #     tf.float32,
    #     shape=(batch_size, RNN_SIZE), name="c1")
    #   h1 = tf.placeholder(
    #     tf.float32,
    #     shape=(batch_size, RNN_SIZE), name="h1")      
    #   return (input_chars, c0, h0, c1, h1)
    else:
      if self.config.mode == "train" or self.config.mode == "eval" or self.config.mode == 'infer':

        dataset = tf.data.Dataset.from_generator(
          generator=lambda: self.get_samples_fn(),
          output_types=(tf.int32, tf.int32))

        if self.config.mode == "train":
          dataset = dataset.shuffle(self.get_num_samples())

        dataset = dataset.repeat(self.config.epochs)

        # Pad all sentences in the same batch to the same length
        dataset = dataset.padded_batch(
          batch_size,
          padded_shapes=([None], [None]))

        dataset = dataset.prefetch(2)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def build(config, augmenter):
  return TextClassificationInputter(config, augmenter)
