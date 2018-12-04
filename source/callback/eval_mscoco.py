"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import os
import numpy as np

import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from callback import Callback

DATASET_DIR = "/mnt/data/data/mscoco"
DATASET_META = "valminusminival2014"

COCO_ID_MAP = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                          25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
                          39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50,
                          51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                          62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76,
                          77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
                          89, 90])


class EvalMSCOCO(Callback):
  def __init__(self, config):
    super(EvalMSCOCO, self).__init__(config)
    self.detection = []
    self.image_ids = []

  def before_run(self, sess):
    self.graph = tf.get_default_graph()

  def after_run(self, sess):
    print("Detection Finished ...")

    for item in self.detection:
      print(item)

    if len(self.detection) > 0:
      annotation_file = os.path.join(
        DATASET_DIR,
        "annotations",
        "instances_" + DATASET_META + ".json")
      coco = COCO(annotation_file)
      coco_results = coco.loadRes(self.detection)
      cocoEval = COCOeval(coco, coco_results, "bbox")
      cocoEval.params.imgIds = self.image_ids
      cocoEval.evaluate()
      cocoEval.accumulate()
      cocoEval.summarize()
    else:
      print("Found no valid detection. Consider re-train your model.")

  def after_step(self, sess, outputs_dict, feed_dict=None):

    num_images = len(outputs_dict["image_id"])
    for i in range(num_images):
      num_detections = len(outputs_dict["labels"][i])
      translation = outputs_dict["translations"][i]
      scale = outputs_dict["scales"][i]

      # COCO evaluation is based on per detection
      for d in range(num_detections):
        box = outputs_dict["bboxes"][i][d]
        box = box - [translation[1], translation[0], translation[1], translation[0]]
        box = box / scale
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        result = {
          "image_id": outputs_dict["image_id"][i],
          "category_id": COCO_ID_MAP[outputs_dict["labels"][i][d]],
          "bbox": box,
          "score": outputs_dict["scores"][i][d]
        }
        self.detection.append(result)
        self.image_ids.append(outputs_dict["image_id"][i])


def build(config):
  return EvalMSCOCO(config)
