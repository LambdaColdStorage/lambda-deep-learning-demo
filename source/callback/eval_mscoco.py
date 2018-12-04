"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import os

import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from callback import Callback

DATASET_DIR = "/mnt/data/data/mscoco"
DATASET_META = "valminusminival2014"


class EvalMSCOCO(Callback):
  def __init__(self, config):
    super(EvalMSCOCO, self).__init__(config)
    self.detection = []
    self.image_ids = []

  def before_run(self, sess):
    self.graph = tf.get_default_graph()

  def after_run(self, sess):
    print("Detection Finished ...")

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

      print(translation)
      print(scale)
      print(outputs_dict["image_id"][i])

      # COCO evaluation is based on per detection
      for d in range(num_detections):
        box = outputs_dict["bboxes"][i][d]
        box = box - [translation[1], translation[0], translation[1], translation[0]]
        box = box / scale
        result = {
          "image_id": outputs_dict["image_id"][i],
          "category_id": outputs_dict["labels"][i][d],
          "bbox": box,
          "score": outputs_dict["scores"][i][d]
        }
        print(result)
        self.detection.append(result)
        self.image_ids.append(outputs_dict["image_id"][i])


def build(config):
  return EvalMSCOCO(config)
