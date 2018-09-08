"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

from callback import Callback


class InferDisplayImageSegmentation(Callback):
  def __init__(self, args):
    super(InferDisplayImageSegmentation, self).__init__(args)
    self.graph = tf.get_default_graph()
    self.colors = np.random.randint(255,
                                    size=(self.args.num_classes, 3))

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

  def after_run(self, sess, saver, summary_writer):
    pass

  def before_step(self, sess):
    pass

  def after_step(self, sess, outputs_dict, saver, summary_writer):
    for p, c in zip(outputs_dict["probabilities"],
                    outputs_dict["classes"]):

      render_label = self.render_label(c,
                                       self.args.num_classes,
                                       self.colors)
      image_out = Image.fromarray(render_label, 'RGB')
      plt.imshow(image_out)
      plt.show()


def build(args):
  return InferDisplayImageSegmentation(args)
