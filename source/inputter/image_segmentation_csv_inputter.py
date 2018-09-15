"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import os
import csv

import tensorflow as tf

from inputter import Inputter
from source.augmenter.external import vgg_preprocessing


class ImageSegmentationCSVInputter(Inputter):
  def __init__(self, config, augmenter):
    super(ImageSegmentationCSVInputter, self).__init__(config, augmenter)

    self.num_samples = -1

    if self.config.mode == "infer":
      self.test_samples = self.config.test_samples

  def get_num_samples(self):
    if self.num_samples < 0:
      if self.config.mode == "infer":
        self.num_samples = len(self.test_samples)
      else:
        with open(self.config.dataset_meta) as f:
          parsed = csv.reader(f, delimiter=",", quotechar="'")
          self.num_samples = len(list(parsed))
    return self.num_samples

  def get_samples_fn(self):
    if self.config.mode == "infer":
      images_path = self.test_samples
      labels_path = self.test_samples
    elif self.config.mode == "train" or \
            self.config.mode == "eval":
      assert os.path.exists(self.config.dataset_meta), (
        "Cannot find dataset_meta file {}.".format(self.config.dataset_meta))

      images_path = []
      labels_path = []
      dirname = os.path.dirname(self.config.dataset_meta)
      with open(self.config.dataset_meta) as f:
        parsed = csv.reader(f, delimiter=",", quotechar="'")
        for row in parsed:
          images_path.append(os.path.join(dirname, row[0]))
          labels_path.append(os.path.join(dirname, row[1]))

    return (images_path, labels_path)

  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)
    max_step = (self.get_num_samples() * self.config.epochs // batch_size)
    tf.constant(max_step, name="max_step")

  def parse_fn(self, image_path, label_path):
    """Parse a single input sample
    """
    image = tf.read_file(image_path)
    image = tf.image.decode_png(image, channels=self.config.image_depth)

    if self.config.mode == "infer":
      image = tf.to_float(image)
      image = vgg_preprocessing._mean_image_subtraction(image)
      label = image[0]
      return image, label
    else:
      label = tf.read_file(label_path)
      label = tf.image.decode_png(label, channels=1)
      label = tf.cast(label, dtype=tf.int64)

      if self.augmenter:
        is_training = (self.config.mode == "train")
        return self.augmenter.augment(image, label,
                                      self.config.output_height,
                                      self.config.output_width,
                                      self.config.resize_side_min,
                                      self.config.resize_side_max,
                                      is_training=is_training,
                                      speed_mode=self.config.augmenter_speed_mode)

  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)

    samples = self.get_samples_fn()

    dataset = tf.data.Dataset.from_tensor_slices(samples)

    if self.config.mode == "train":
      dataset = dataset.shuffle(self.config.shuffle_buffer_size)

    dataset = dataset.repeat(self.config.epochs)

    dataset = dataset.map(
      lambda image, label: self.parse_fn(image, label),
      num_parallel_calls=4)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(2)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def build(config, augmenter):
  return ImageSegmentationCSVInputter(config, augmenter)
