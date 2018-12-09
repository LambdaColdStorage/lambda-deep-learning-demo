"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import misc
from scipy import ndimage

import tensorflow as tf

from .callback import Callback

MSCOCO_CAT_NAME = [u'person', u'bicycle', u'car', u'motorcycle', u'airplane',
                   u'bus', u'train', u'truck', u'boat', u'traffic light',
                   u'fire hydrant', u'stop sign', u'parking meter', u'bench',
                   u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow',
                   u'elephant', u'bear', u'zebra', u'giraffe', u'backpack',
                   u'umbrella', u'handbag', u'tie', u'suitcase', u'frisbee',
                   u'skis', u'snowboard', u'sports ball', u'kite',
                   u'baseball bat', u'baseball glove', u'skateboard',
                   u'surfboard', u'tennis racket', u'bottle', u'wine glass',
                   u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana',
                   u'apple', u'sandwich', u'orange', u'broccoli', u'carrot',
                   u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch',
                   u'potted plant', u'bed', u'dining table', u'toilet', u'tv',
                   u'laptop', u'mouse', u'remote', u'keyboard', u'cell phone',
                   u'microwave', u'oven', u'toaster', u'sink', u'refrigerator',
                   u'book', u'clock', u'vase', u'scissors', u'teddy bear',
                   u'hair drier', u'toothbrush']


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7


class InferDisplayObjectDetection(Callback):
  def __init__(self, config):
    super(InferDisplayObjectDetection, self).__init__(config)

  def before_run(self, sess):
    self.graph = tf.get_default_graph()
    self.RGB_MEAN = [123.68, 116.78, 103.94]

  def display_ori(self, outputs_dict):
    for s, l, b, a, scale, translation, file_name in zip(
      outputs_dict["scores"],
      outputs_dict["labels"],
      outputs_dict["bboxes"],
      outputs_dict["anchors"],
      outputs_dict["scales"],
      outputs_dict["translations"],
      outputs_dict["file_name"]):

      input_image = misc.imread(file_name).astype(np.float32) / 255.0

      plt.figure()
      plt.axis('off')

      for label, box in zip(l, b):
        # Compute the location to draw annotation
        box = box - [translation[1], translation[0], translation[1], translation[0]]
        box = box / scale
        print(box)
        label = MSCOCO_CAT_NAME[label - 1]
        ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        top_left = [box[0] + 1, box[1] - 1.3 * lineh]
        if top_left[1] < 0:     # out of image
            top_left[1] = box[3] - 1.3 * lineh

        # Draw twice to make detections more visually noticable
        cv2.rectangle(input_image, (box[0], box[1]), (box[2], box[3]),
                      color=(0, 0, 0), thickness=5)
        cv2.rectangle(input_image, (box[0], box[1]), (box[2], box[3]),
                      color=(1, 0, 0), thickness=2)

        cv2.putText(input_image, label,
                    (int(top_left[0]), int(top_left[1] + lineh)),
                    FONT, FONT_SCALE, color=(0, 0, 0),
                    lineType=cv2.LINE_AA, thickness=3)
        cv2.putText(input_image, label,
                    (int(top_left[0]), int(top_left[1] + lineh)),
                    FONT, FONT_SCALE, color=(1, 1, 0),
                    lineType=cv2.LINE_AA, thickness=1)

      plt.imshow(input_image)
      plt.show()

  def display_normalized(self, outputs_dict):
    for input_image, s, l, b, a, scale, translation, file_name in zip(
      outputs_dict["images"],
      outputs_dict["scores"],
      outputs_dict["labels"],
      outputs_dict["bboxes"],
      outputs_dict["anchors"],
      outputs_dict["scales"],
      outputs_dict["translations"],
      outputs_dict["file_name"]):

      input_image = (input_image + self.RGB_MEAN).astype(np.float32)
      input_image = np.clip(input_image, 0, 255) / 255.0

      plt.figure()
      plt.axis('off')

      for label, box, anchor in zip(l, b, a):
        # Compute the location to draw annotation
        print("box: ")
        print(box)
        label = MSCOCO_CAT_NAME[label - 1]
        ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        top_left = [box[0] + 1, box[1] - 1.3 * lineh]
        if top_left[1] < 0:     # out of image
            top_left[1] = box[3] - 1.3 * lineh

        # Draw twice to make detections more visually noticable
        cv2.rectangle(input_image, (box[0], box[1]), (box[2], box[3]),
                      color=(0, 0, 0), thickness=5)
        cv2.rectangle(input_image, (box[0], box[1]), (box[2], box[3]),
                      color=(1, 0, 0), thickness=2)

        # # Draw anchor for debugging
        # anchor = np.clip(anchor, 0, input_image.shape[0])
        # cv2.rectangle(input_image, (anchor[0], anchor[1]), (anchor[2], anchor[3]),
        #               color=(0, 1, 0), thickness=2)
        print("anchor: ")
        print(anchor)

        cv2.putText(input_image, label,
                    (int(top_left[0]), int(top_left[1] + lineh)),
                    FONT, FONT_SCALE, color=(0, 0, 0),
                    lineType=cv2.LINE_AA, thickness=3)
        cv2.putText(input_image, label,
                    (int(top_left[0]), int(top_left[1] + lineh)),
                    FONT, FONT_SCALE, color=(1, 1, 0),
                    lineType=cv2.LINE_AA, thickness=1)

      plt.imshow(input_image)
      plt.show()

  def after_step(self, sess, outputs_dict, feed_dict=None):
    self.display_normalized(outputs_dict)


def build(config):
  return InferDisplayObjectDetection(config)
