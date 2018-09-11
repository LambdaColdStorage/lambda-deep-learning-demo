from augmenter.external import cifarnet_preprocessing


def augment(image, output_height, output_width, is_training=False,
            add_image_summaries=False, speed_mode=False):
  return cifarnet_preprocessing.preprocess_image(
    image, output_height, output_width, is_training, add_image_summaries, speed_mode=speed_mode)
