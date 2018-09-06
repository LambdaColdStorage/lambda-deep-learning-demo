import tensorflow as tf

from augmenter.external import vgg_preprocessing

def augment(image, label, output_height, output_width,
            resize_side_min, resize_side_max, is_training=False):
  if is_training:
    resize_side = tf.random_uniform(
      [],
      minval=resize_side_min,
      maxval=resize_side_max + 1,
      dtype=tf.int32)

    image = vgg_preprocessing._aspect_preserving_resize(
      image, resize_side, 3, 'bilinear')
    image = vgg_preprocessing._central_crop(
      [image], output_height, output_width)[0]
    image = tf.reshape(image, [output_height, output_width, 3])

    label = vgg_preprocessing._aspect_preserving_resize(
      label, resize_side, 1, 'nearest')
    label = vgg_preprocessing._central_crop(
      [label], output_height, output_width)[0]
    label = tf.reshape(label, [output_height, output_width])
  else:
    image = tf.image.resize_images(
      image, [output_height, output_width],
      tf.image.ResizeMethod.BILINEAR)
    image = tf.reshape(image, [output_height, output_width, 3])

    label = tf.image.resize_images(
      label, [output_height, output_width],
      tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.reshape(label, [output_height, output_width])

  image = image - 0.5
  return image, label
