import tensorflow as tf

from augmenter.external import vgg_preprocessing


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min,
                         resize_side_max,
                         speed_mode=False):
  if speed_mode:
    image = tf.image.resize_images(
      image, [output_height, output_width],
      tf.image.ResizeMethod.BILINEAR)
    image = tf.reshape(image, [output_height, output_width, 3])
    image = tf.to_float(image)
    image = vgg_preprocessing._mean_image_subtraction(image)
  else:
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
    image = tf.to_float(image)
    image = vgg_preprocessing._mean_image_subtraction(image)

  return image


def preprocess_for_eval(image,
                        output_height,
                        output_width,
                        resize_side,
                        speed_mode=False):
  if speed_mode:
    image = tf.image.resize_images(
      image, [output_height, output_width],
      tf.image.ResizeMethod.BILINEAR)
    image = tf.reshape(image, [output_height, output_width, 3])
    image = tf.to_float(image)
    image = vgg_preprocessing._mean_image_subtraction(image)
  else:
    resize_side = resize_side
    image = vgg_preprocessing._aspect_preserving_resize(
      image, resize_side, 3, 'bilinear')
    image = vgg_preprocessing._central_crop(
      [image], output_height, output_width)[0]
    image = tf.reshape(image, [output_height, output_width, 3])
    image = tf.to_float(image)
    image = vgg_preprocessing._mean_image_subtraction(image)

  return image


def augment(image, output_height, output_width,
            resize_side_min, resize_side_max,
            is_training=False, speed_mode=False):
  if is_training:
    return preprocess_for_train(image,
                                output_height,
                                output_width,
                                resize_side_min,
                                resize_side_max,
                                speed_mode=speed_mode)
  else:
    return preprocess_for_eval(image,
                               output_height,
                               output_width,
                               resize_side_min,
                               speed_mode=speed_mode)
