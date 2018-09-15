"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest

from inputter import Inputter


class ImageClassificationSynInputter(Inputter):
  def __init__(self, config, augmenter):
    super(ImageClassificationSynInputter, self).__init__(config, augmenter)

    self.num_samples = 256

  def get_num_samples(self):
    return self.num_samples

  def get_samples_fn(self):
    pass

  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)
    max_step = (self.get_num_samples() * self.config.epochs // batch_size)
    tf.constant(max_step, name="max_step")

  def parse_fn(self, image, label):

    if self.augmenter:
      is_training = (self.config.mode == "train")
      image = self.augmenter.augment(image,
                                     self.config.image_height,
                                     self.config.image_width,
                                     is_training=is_training,
                                     speed_mode=self.config.augmenter_speed_mode)
    return (image, label)

  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)

    image_dtype = tf.float32
    label_dtype = tf.int32
    image_value = 0.0
    label_value = 0

    shape = ([self.get_num_samples(), self.config.image_height,
              self.config.image_width, self.config.image_depth])

    image_shape = tf.TensorShape(shape)
    label_shape = tf.TensorShape([self.get_num_samples(),
                                  self.config.num_classes])

    image_element = nest.map_structure(
        lambda s: tf.constant(image_value, image_dtype, s), image_shape)

    label_element = nest.map_structure(
        lambda s: tf.constant(label_value, label_dtype, s), label_shape)

    dataset = tf.data.Dataset.from_tensor_slices(
      (image_element, label_element)).repeat(self.config.epochs)

    dataset = dataset.map(
      lambda image, label: self.parse_fn(image, label),
      num_parallel_calls=4)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(2)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def build(config, augmenter):
  return ImageClassificationSynInputter(config, augmenter)
