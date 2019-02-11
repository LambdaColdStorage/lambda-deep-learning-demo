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

import tensorflow as tf

from .inputter import Inputter


RNN_SIZE = 256

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


def buildVocab(embedding_filename, dataset_meta):
  chars, embd = loadGloVe(embedding_filename)
  vocab_size = len(chars)
  embedding_dim = len(embd[0])
  embedding = np.asarray(embd).astype(np.float32)

  char2idx = {}
  for i, v in enumerate(chars):
    char2idx[v] = i

  data = []
  for meta in dataset_meta:
    with open(meta, 'rb') as f:
      d = f.read()
  d = d.lower()
  data = d.split()

  whole_seq = np.array([char2idx[c] for c in data if c in char2idx], dtype='int32')

  return chars, vocab_size, embedding, char2idx, whole_seq


def buildVocabFast(embedding_filename, dataset_meta):

  data = []
  for meta in dataset_meta:
    with open(meta, 'rb') as f:
      d = f.read()
  d = d.lower()
  data = d.split()

  counter = Counter(data)
  char_cnt = sorted(counter.items(),
                    key=operator.itemgetter(1), reverse=True)
  chars_from_data = [x[0] for x in char_cnt]


  chars_from_pretrain, embd = loadGloVe(embedding_filename)
  vocab_size = len(chars_from_pretrain)
  embedding_dim = len(embd[0])
  embedding = np.asarray(embd).astype(np.float32)

  char2idx_pretrain = {}
  for i, v in enumerate(chars_from_pretrain):
    char2idx_pretrain[v] = i

  idx = np.array([char2idx_pretrain[item] if item in char2idx_pretrain else -1 for item in chars_from_data], dtype='int32')

  mask = idx >= 0
  idx = idx[mask]
  chars = [chars_from_data[i] for i, m in enumerate(mask) if m]
  vocab_size = len(chars)
  embedding = embedding[idx, :]

  char2idx = {}
  for i, v in enumerate(chars):
    char2idx[v] = i

  whole_seq = np.array([char2idx[c] for c in data if c in char2idx], dtype='int32')
  whole_seq_char = [c for c in data if c in char2idx]

  return chars, vocab_size, embedding, char2idx, whole_seq

class TextGenerationWord2VecInputter(Inputter):
  def __init__(self, config, augmenter):
    super(TextGenerationWord2VecInputter, self).__init__(config, augmenter)

    if self.config.mode == "train":
      self.num_samples = 100000
      self.seq_length = 50
    elif self.config.mode == "infer":
      self.num_samples = 1000
      self.seq_length = 1
    elif self.config.mode == "eval":
      self.num_samples = 10000
      self.seq_length = 50
    elif self.config.mode == "export":
      self.num_samples = 1
      self.seq_length = 1

    self.embedding_filename = '/home/chuan/Downloads/glove.6B/glove.6B.50d.txt'

    self.chars, self.vocab_size, self.embedding, self.char2idx, self.whole_seq = buildVocabFast(
      self.embedding_filename, self.config.dataset_meta)

    # self.initial_seq()


  # def initial_seq(self):

  #   data = []
  #   for meta in self.config.dataset_meta:
  #     with open(meta, 'rb') as f:
  #       d = f.read()
  #   data = d.split()

  #   self.whole_seq = np.array([self.char2idx[c] for c in data if c in self.char2idx], dtype='int32')
  #   print(self.char2idx['='])


  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)
    max_step = (self.get_num_samples() * self.config.epochs // batch_size)
    tf.constant(max_step, name="max_step")

  def get_num_samples(self):
    return self.num_samples

  def get_vocab_size(self):
    return self.vocab_size

  def get_chars(self):
    return self.chars, self.embedding

  def get_seq_length(self):
    return self.seq_length

  def get_samples_fn(self):
    random_starts = np.random.randint(
      0,
      self.whole_seq.shape[0] - self.seq_length - 1,
      (self.num_samples,))

    for st in random_starts:
        seq = self.whole_seq[st:st + self.seq_length + 1]
        yield seq[:-1], seq[1:]

  def parse_fn(self, inputs, outputs):
    """Parse a single input sample
    """
    inputs.set_shape([self.seq_length])
    outputs.set_shape([self.seq_length])

    return (inputs, outputs)

  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count) 
    if self.config.mode == "export":
      input_chars = tf.placeholder(tf.int32,
                             shape=(batch_size, self.seq_length),
                             name="input_chars")
      c0 = tf.placeholder(
        tf.float32,
        shape=(batch_size, RNN_SIZE), name="c0")
      h0 = tf.placeholder(
        tf.float32,
        shape=(batch_size, RNN_SIZE), name="h0")
      c1 = tf.placeholder(
        tf.float32,
        shape=(batch_size, RNN_SIZE), name="c1")
      h1 = tf.placeholder(
        tf.float32,
        shape=(batch_size, RNN_SIZE), name="h1")      
      return (input_chars, c0, h0, c1, h1)
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
        return (tf.zeros([batch_size, self.seq_length], tf.int32),
                tf.zeros([batch_size, self.seq_length], tf.int32))


def build(config, augmenter):
  return TextGenerationWord2VecInputter(config, augmenter)
