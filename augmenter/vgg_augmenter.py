from augmenter.external import vgg_preprocessing


def augment(image, output_height, output_width, is_training=False):
  return vgg_preprocessing.preprocess_image(
    image, output_height, output_width, is_training)
