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
    # print(outputs_dict["scores"][0].shape)
    for s, b, input_image in zip(outputs_dict["scores"],
                         outputs_dict["bboxes"],
                         outputs_dict["images"]):

      input_image = input_image + self.RGB_MEAN
      input_image = np.clip(input_image, 0, 255)
      input_image = input_image / 255.0

      plt.figure()
      plt.axis('off')
      for box in b:
        cv2.rectangle(input_image, (box[0], box[1]), (box[2], box[3]),
                      color=(255, 255, 55), thickness=2)
      plt.imshow(input_image)
      plt.show()

def build(config):
  return InferDisplayObjectDetection(config)
