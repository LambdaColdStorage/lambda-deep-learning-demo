from source.augmenter.external import inception_preprocessing


def augment(image, output_height, output_width, is_training=False, speed_mode=False):
  return inception_preprocessing.preprocess_image(
    image, output_height, output_width, is_training, fast_mode=speed_mode)
