import numpy as np
import cv2


import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.

RANDOM_SUB_SAMPLE = False
RANDOM_HFLIP = True
RANDOM_COLOR = True

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


def safe_divide(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(name, 'bboxes_resize_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_resize(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def bboxes_intersection(bbox_ref, bboxes, name=None):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.
    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores


def bboxes_filter_overlap(labels, bboxes,
                          threshold=0.5, assign_negative=False,
                          scope=None):
    """Filter out bounding boxes based on (relative )overlap with reference
    box [0, 0, 1, 1].  Remove completely bounding boxes, or assign negative
    labels to the one outside (useful for latter processing...).
    Return:
      labels, bboxes: Filtered (or newly assigned) elements.
    """
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                     bboxes)
        mask = scores > threshold
        mask.set_shape([None])
        if assign_negative:
            labels = tf.where(mask, labels, -labels)
            # bboxes = tf.where(mask, bboxes, bboxes)
        else:
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes

def fix_image_flip_shape(image, result):
    """Set the shape to 3 dimensional if we don't know anything else.
    Args:
      image: original image size
      result: flipped or transformed image
    Returns:
      An image whose shape is at least None,None,None.
    """
    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result

def random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes.
    """
    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        image_shape = tf.to_float(tf.shape(image))
        bboxes = tf.stack([image_shape[1] - bboxes[:, 2], bboxes[:, 1],
                           image_shape[1] - bboxes[:, 0], bboxes[:, 3]], axis=-1)        
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):
        image = tf.convert_to_tensor(image, name='image')
        uniform_random = tf.random.uniform([], 0, 1.0, seed=seed)
        mirror_cond = tf.less(uniform_random, .5)
        # Flip image.
        image = tf.cond(mirror_cond,
                        lambda: tf.image.flip_left_right(image),
                        lambda: image)
        # Flip bboxes.
        bboxes = tf.cond(mirror_cond,
                         lambda: flip_bboxes(bboxes),
                         lambda: bboxes)
        return image, bboxes

def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.
    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image,
                         classes,
                         boxes,
                         resolution,
                         speed_mode=False):
  if speed_mode:
    pass
  else:

    image_ori = image
    classes_ori = classes
    boxes_ori = boxes

    # randomly sample patches
    if RANDOM_SUB_SAMPLE:
      image_shape = tf.shape(image)

      x1, y1, x2, y2 = tf.unstack(boxes, 4, axis=1)
      x1 = tf.expand_dims(tf.div(x1, tf.to_float(image_shape[1])), -1)
      x2 = tf.expand_dims(tf.div(x2, tf.to_float(image_shape[1])), -1)
      y1 = tf.expand_dims(tf.div(y1, tf.to_float(image_shape[0])), -1)
      y2 = tf.expand_dims(tf.div(y2, tf.to_float(image_shape[0])), -1)
      boxes = tf.concat([y1, x1, y2, x2], axis=1)

      bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        image_shape,
        bounding_boxes=tf.expand_dims(boxes, 0),
        min_object_covered=BBOX_CROP_OVERLAP,
        aspect_ratio_range=(0.8, 1.2),
        area_range=(0.5, 1.0),
        max_attempts=200,
        use_image_if_no_bounding_boxes=True)

      distort_bbox = distort_bbox[0, 0]

      image = tf.slice(image, bbox_begin, bbox_size)

      boxes = bboxes_resize(distort_bbox, boxes)

      classes, boxes = bboxes_filter_overlap(classes, boxes,
                                             threshold=BBOX_CROP_OVERLAP,
                                             assign_negative=False)

      y1, x1, y2, x2 = tf.unstack(boxes, 4, axis=1)
      image_shape = tf.shape(image)

      w = tf.to_float(image_shape[1])
      h = tf.to_float(image_shape[0])

      x1 = tf.scalar_mul(w, x1)
      x2 = tf.scalar_mul(w, x2)
      y1 = tf.scalar_mul(h, y1)
      y2 = tf.scalar_mul(h, y2)

      x1 = tf.clip_by_value(x1, 0.0, w)
      x2 = tf.clip_by_value(x2, 0.0, w)
      y1 = tf.clip_by_value(y1, 0.0, h)
      y2 = tf.clip_by_value(y2, 0.0, h)

      x1 = tf.expand_dims(x1, -1)
      x2 = tf.expand_dims(x2, -1)
      y1 = tf.expand_dims(y1, -1)
      y2 = tf.expand_dims(y2, -1)

      boxes = tf.concat([x1, y1, x2, y2], axis=1)

    if RANDOM_HFLIP:
      # Randomly flip the image horizontally.
      image, boxes = random_flip_left_right(image, boxes)
    
    if RANDOM_COLOR:
      # Color distortion
      image = tf.div(image, 255.0)
      image = apply_with_random_selector(
              image,
              lambda x, ordering: distort_color(x, ordering, fast_mode=False),
              num_cases=4)
      image = tf.scalar_mul(255.0, image)
    
    # Rollback to the full image if there is no boxes
    no_box_cond = tf.equal(tf.size(boxes), 0)
    image = tf.cond(no_box_cond,
                    lambda: image_ori,
                    lambda: image)
    boxes = tf.cond(no_box_cond,
                    lambda: boxes_ori,
                    lambda: boxes)
    classes = tf.cond(no_box_cond,
                     lambda: classes_ori,
                     lambda: classes)

    # transform image and boxes
    image, scale, translation = aspect_preserving_resize(image, resolution, depth=3, resize_mode="bilinear")
    image = tf.image.resize_image_with_crop_or_pad(
      image,
      resolution,
      resolution)
    boxes = tf.scalar_mul(scale, boxes) + [translation[1], translation[0], translation[1], translation[0]]

    # mean subtraction
    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    channels = tf.split(axis=2, num_or_size_splits=3, value=image)
    for i in range(3):
      channels[i] -= means[i]
    image = tf.concat(axis=2, values=channels)

  return image, classes, boxes, scale, translation


def preprocess_for_eval(image,
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

  return image, classes, boxes, scale, translation


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

