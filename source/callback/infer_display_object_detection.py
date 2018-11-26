"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf

from callback import Callback


class InferDisplayObjectDetection(Callback):
  def __init__(self, config):
    super(InferDisplayObjectDetection, self).__init__(config)

  def before_run(self, sess):
    self.graph = tf.get_default_graph()
    self.RGB_MEAN = [123.68, 116.78, 103.94]

  def after_step(self, sess, outputs_dict, feed_dict=None):

    for s, l, b, a, input_image in zip(outputs_dict["scores"],
                         outputs_dict["labels"],
                         outputs_dict["bboxes"],
                         outputs_dict["anchors"],
                         outputs_dict["images"]):

      input_image = input_image + self.RGB_MEAN
      input_image = np.clip(input_image, 0, 255)
      input_image = input_image / 255.0

      print(s)
      print(l)
      print(b)
      print(a)

      plt.figure()
      plt.axis('off')
      for box in b:
        cv2.rectangle(input_image, (box[0], box[1]), (box[2], box[3]),
                      color=(1, 0, 0), thickness=2)
      for anchor in a:
        cv2.rectangle(input_image, (anchor[0], anchor[1]), (anchor[2], anchor[3]),
                      color=(0, 1, 0), thickness=2)        
      plt.imshow(input_image)
      plt.show()


def build(config):
  return InferDisplayObjectDetection(config)
