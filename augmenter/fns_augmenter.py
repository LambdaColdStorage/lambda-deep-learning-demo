import tensorflow as tf

from augmenter.external import vgg_preprocessing


def augment(image, output_height, output_width,
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

  image = image * 255.0
  image = vgg_preprocessing._mean_image_subtraction(image)
  return image
