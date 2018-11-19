import numpy as np
import cv2


import tensorflow as tf


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def compute_new_shape(height, width, resolution):
  height = tf.to_float(height)
  width = tf.to_float(width)
  resolution = tf.to_float(resolution)

  scale = tf.cond(tf.greater(height, width),
                  lambda: resolution / height,
                  lambda: resolution / width)
  new_height = tf.to_int32(tf.rint(height * scale))
  new_width = tf.to_int32(tf.rint(width * scale))

  translation = (resolution - [tf.to_float(new_height), tf.to_float(new_width)]) / 2.0
  return new_height, new_width, scale, translation


def aspect_preserving_resize(image, resolution, depth=3, resize_mode="bilinear"):
  resolution = tf.convert_to_tensor(resolution, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]

  new_height, new_width, scale, translation = compute_new_shape(height, width, resolution)
  image = tf.expand_dims(image, 0)

  if resize_mode == 'bilinear':
    resized_image = tf.image.resize_bilinear(image,
                                             [new_height, new_width],
                                             align_corners=False)
  elif resize_mode == 'nearest':
    resized_image = tf.image.resize_nearest_neighbor(image,
                                                     [new_height, new_width],
                                                     align_corners=False)
  else:
    assert False, "Unknown image resize mode: '{}'".format(resize_mode)

  resized_image = tf.squeeze(resized_image, 0)
  new_shape = tf.shape(resized_image)

  resized_image.set_shape([None, None, depth])

  return resized_image, scale, translation



def preprocess_for_train(image,
                         classes,
                         boxes,
                         resolution,
                         speed_mode=False):
  if speed_mode:
    pass
  else:
    # mean subtraction
    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    channels = tf.split(axis=2, num_or_size_splits=3, value=image)
    for i in range(3):
      channels[i] -= means[i]
    image = tf.concat(axis=2, values=channels)

    # transform image and boxes
    image, scale, translation = aspect_preserving_resize(image, resolution, depth=3, resize_mode="bilinear")
    image = tf.image.resize_image_with_crop_or_pad(
      image,
      resolution,
      resolution)
    boxes = tf.scalar_mul(scale, boxes) + [translation[1], translation[0], translation[1], translation[0]]

  return image, classes, boxes


def preprocess_for_eval(image,
                        classes,
                        boxes,
                        resolution,
                        speed_mode=False):
  if speed_mode:
    pass
  else:
    pass

  return image, classes, boxes


def augment(image, classes, boxes, resolution,
            is_training=False, speed_mode=False):
  if is_training:
    return preprocess_for_train(image,
                                classes,
                                boxes,
                                resolution,
                                speed_mode=speed_mode)
  else:
    return preprocess_for_eval(image,
                               classes,
                               boxes,
                               resolution,
                               speed_mode=speed_mode)

