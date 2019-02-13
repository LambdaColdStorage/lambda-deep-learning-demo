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


def loadSentences(dataset_meta):
  sentences = []
  labels = []
  for meta in dataset_meta:
    dirname = os.path.dirname(meta)
    with open(meta) as f:
      parsed = csv.reader(f, delimiter="\t")
      for row in parsed:
        sentences.append(row[0].split(" "))
        # sentences.append(re.findall(r"[\w']+|[.,!?;]", row[3]))
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


def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    file.close()
    return vocab,embd


def fetchPretrain(embedding_filename, words2idx, words):

  words_from_pretrain, embd = loadGloVe(embedding_filename)

  embedding = np.asarray(embd).astype(np.float32)

  words2idx_pretrain = {}
  for i, v in enumerate(words_from_pretrain):
    words2idx_pretrain[v] = i

  idx = np.array([words2idx_pretrain[item] if item in words2idx_pretrain else -1 for item in words], dtype='int32')
  mask = idx >= 0
  idx = idx[mask]
  words = [words[i] for i, m in enumerate(mask) if m]
  embedding = embedding[idx, :]

  words2idx = {}
  for i, v in enumerate(words):
    words2idx[v] = i

  return embedding, words2idx, words

class TextClassificationPretrainInputter(Inputter):
  def __init__(self, config, augmenter):
    super(TextClassificationPretrainInputter, self).__init__(config, augmenter)

    f = open(self.config.vocab_file, 'rb')
    self.words2idx, self.words = pickle.load(f)
    f.close()

    self.embedding_filename = '/home/ubuntu/demo/model/glove.6B/glove.6B.50d.txt'

    self.embedding, self.words2idx, self.words = fetchPretrain(self.embedding_filename, self.words2idx, self.words)

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

  def get_embedding(self):
    return self.embedding

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
    else:
      if self.config.mode == "train" or self.config.mode == "eval" or self.config.mode == 'infer':

        dataset = tf.data.Dataset.from_generator(
          generator=lambda: self.get_samples_fn(),
          output_types=(tf.int32, tf.int32))

        if self.config.mode == "train":
          dataset = dataset.shuffle(self.get_num_samples())

        # dataset = dataset.shuffle(self.get_num_samples())

        dataset = dataset.repeat(self.config.epochs)

        # Pad all sentences in the same batch to the same length
        dataset = dataset.padded_batch(
          batch_size,
          padded_shapes=([None], [None]),
          padding_values=(-1, -1))

        dataset = dataset.prefetch(2)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def build(config, augmenter):
  return TextClassificationPretrainInputter(config, augmenter)
