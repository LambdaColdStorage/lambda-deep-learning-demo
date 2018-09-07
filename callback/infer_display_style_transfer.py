"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from callback import Callback


class InferDisplayStyleTransfer(Callback):
  def __init__(self, args):
    super(InferDisplayStyleTransfer, self).__init__(args)
    self.graph = tf.get_default_graph()
    self.RGB_MEAN = [123.68, 116.78, 103.94]

  def render_label(self, label, num_classes, label_colors):

    label = label.astype(int)
    r = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    g = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    b = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)

    for i_color in range(0, num_classes):
      r[label == i_color] = label_colors[i_color, 0]
      g[label == i_color] = label_colors[i_color, 1]
      b[label == i_color] = label_colors[i_color, 2]

    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

  def before_run(self, sess, saver):
    pass

  def after_run(self, sess, saver):
    pass

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver):
    for input_image, output_image in zip(
      outputs_dict["input"], outputs_dict["output"]):

        input_image = input_image + self.RGB_MEAN
        input_image = np.clip(input_image, 0, 255)
        input_image = input_image.astype(np.uint8)

        transformed_image = np.clip(output_image, 0, 255)
        transformed_image = transformed_image.astype(np.uint8)

        w = min(input_image.shape[1], transformed_image.shape[1])
        h = min(input_image.shape[0], transformed_image.shape[0])
        display_image = np.concatenate(
          (input_image[0:h, 0:w, :], transformed_image[0:h, 0:w, :]), axis=1)

        plt.figure()
        plt.axis('off')
        plt.imshow(display_image)
        plt.show()


def build(args):
  return InferDisplayStyleTransfer(args)
