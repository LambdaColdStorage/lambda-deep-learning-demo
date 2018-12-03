"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import pickle

import tensorflow as tf

from callback import Callback


class EvalMSCOCO(Callback):
  def __init__(self, config):
    super(EvalMSCOCO, self).__init__(config)
    self.detection = []
    self.image_ids = []

  def before_run(self, sess):
    self.graph = tf.get_default_graph()

  def after_run(self, sess):
    pickle.dump(
      (self.detection, self.image_ids), open("detection_results.p", "wb"))

  def after_step(self, sess, outputs_dict, feed_dict=None):

    num_images = len(outputs_dict["image_id"])
    # print('--------------------------------------------------')
    # print(len(outputs_dict["image_id"]))
    # print('--------------------------------------------------')
    # print(len(outputs_dict["labels"]))
    # print('--------------------------------------------------')
    # print(len(outputs_dict["bboxes"]))
    # print('--------------------------------------------------')
    # print(len(outputs_dict["scores"]))
    for i in range(num_images):
      result = {
        "image_id": outputs_dict["image_id"][i],
        "category_id": outputs_dict["labels"][i],
        "bbox": outputs_dict["bboxes"][i],
        "score": outputs_dict["scores"][i]
      }

      self.detection.append(result)
      self.image_ids.append(outputs_dict["image_id"][i])


def build(config):
  return EvalMSCOCO(config)
