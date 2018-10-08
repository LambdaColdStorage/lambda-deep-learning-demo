"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

from inputter import Inputter
from pycocotools.coco import COCO


JSON_TO_IMAGE = {
    "train2014": "train2014",
    "val2014": "val2014",
    "valminusminival2014": "val2014",
    "minival2014": "val2014",
    "test2014": "test2014"
}


class ObjectDetectionMSCOCOInputter(Inputter):
  def __init__(self, config, augmenter):
    super(ObjectDetectionMSCOCOInputter, self).__init__(config, augmenter)

    self.num_samples = -1
    self.category_id_to_class_id = None
    self.class_id_to_category_id = None
    self.cat_names = None
    
    self.anchors = None
    self.anchors_stride = 16
    self.anchors_sizes = (32, 64, 128, 256, 512)
    self.anchors_aspect_ratios = (0.5, 1.0, 2.0)

  def get_num_samples(self):
    return self.num_samples

  def get_anchors(self):
    if not self.anchors:
      self.generate_anchors()
    return self.anchors

  def get_samples_fn(self):

    # Read image
    samples = []
    for name_meta in self.config.dataset_meta:
      annotation_file = os.path.join(
        self.config.dataset_dir,
        "annotations",
        "instances_" + name_meta + ".json")
      coco = COCO(annotation_file)

      cat_ids = coco.getCatIds()
      self.cat_names = [c["name"] for c in coco.loadCats(cat_ids)]

      # background has class id of 0
      self.category_id_to_class_id = {
        v: i + 1 for i, v in enumerate(cat_ids)}
      self.class_id_to_category_id = {
        v: k for k, v in self.category_id_to_class_id.items()}

      img_ids = coco.getImgIds()
      img_ids.sort()
      # list of dict, each has keys: height,width,id,file_name
      imgs = coco.loadImgs(img_ids)

      for img in imgs:
        img["file_name"] = os.path.join(
          self.config.dataset_dir,
          JSON_TO_IMAGE[name_meta],
          img["file_name"])

        self.parse_gt(coco, self.category_id_to_class_id, img)

      samples.extend(imgs) 

    # Filter out images that has no object.
    num = len(samples)

    samples = list(filter(
      lambda sample: len(
        sample['boxes'][sample['is_crowd'] == 0]) > 0, samples))

    for sample in samples:
      if 'class' in sample:
        sample['class'] = sample['class'].tolist()
      if 'boxes' in sample:
        sample['boxes'] = sample['boxes'].tolist()
      if 'is_crowd' in sample:
        sample['is_crowd'] = sample['is_crowd'].tolist()

      yield (sample["file_name"],
             sample["class"],
             sample["boxes"],
             sample["is_crowd"])

  def parse_gt(self, coco, category_id_to_class_id, img):
    ann_ids = coco.getAnnIds(imgIds=img["id"], iscrowd=None)
    objs = coco.loadAnns(ann_ids)  

    # clean-up boxes
    valid_objs = []
    width = img["width"]
    height = img["height"]

    for obj in objs:
      if obj.get("ignore", 0) == 1:
          continue
      x1, y1, w, h = obj["bbox"]

      x1 = float(x1)
      y1 = float(y1)
      x2 = float(x1 + w)
      y2 = float(y1 + h)

      x1 = max(0, min(float(x1), width - 1))
      y1 = max(0, min(float(y1), height - 1))
      x2 = max(0, min(float(x2), width - 1))
      y2 = max(0, min(float(y2), height - 1))

      w = x2 - x1
      h = y2 - y1

      if obj['area'] > 1 and w > 0 and h > 0 and w * h >= 4:
        obj['bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)

    boxes = np.asarray([obj['bbox'] for obj in valid_objs], dtype='float32')  # (n, 4)

    cls = np.asarray([
        category_id_to_class_id[obj['category_id']]
        for obj in valid_objs], dtype='int32')  # (n,)

    is_crowd = np.asarray([obj['iscrowd'] for obj in valid_objs], dtype='int8')

    img['boxes'] = boxes # nx4
    img['class'] = cls # n, always >0
    img['is_crowd'] = is_crowd # n,

  def _whctrs(self, anchor):
      """Return width, height, x center, and y center for an anchor (window)."""
      w = anchor[2] - anchor[0] + 1
      h = anchor[3] - anchor[1] + 1
      x_ctr = anchor[0] + 0.5 * (w - 1)
      y_ctr = anchor[1] + 0.5 * (h - 1)
      return w, h, x_ctr, y_ctr

  def _mkanchors(self, ws, hs, x_ctr, y_ctr):
      """Given a vector of widths (ws) and heights (hs) around a center
      (x_ctr, y_ctr), output a set of anchors (windows).
      """
      ws = ws[:, np.newaxis]
      hs = hs[:, np.newaxis]
      anchors = np.hstack(
          (
              x_ctr - 0.5 * (ws - 1),
              y_ctr - 0.5 * (hs - 1),
              x_ctr + 0.5 * (ws - 1),
              y_ctr + 0.5 * (hs - 1)
          )
      )
      return anchors

  def _ratio_enum(self, anchor, ratios):
      """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
      w, h, x_ctr, y_ctr = self._whctrs(anchor)
      size = w * h
      size_ratios = size / ratios
      ws = np.round(np.sqrt(size_ratios))
      hs = np.round(ws * ratios)
      anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
      return anchors

  def _scale_enum(self, anchor, scales):
      """Enumerate a set of anchors for each scale wrt an anchor."""
      w, h, x_ctr, y_ctr = self._whctrs(anchor)     
      ws = w * scales
      hs = h * scales
      anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
      return anchors

  def generate_anchors(self):
    anchor = np.array(
      [1, 1, self.anchors_stride, self.anchors_stride], dtype=np.float) - 1
    anchors = self._ratio_enum(
      anchor, np.array(self.anchors_aspect_ratios, dtype=np.float))
    anchors = np.vstack(
        [self._scale_enum(
         anchors[i, :],
         np.array(self.anchors_sizes, dtype=np.float) / self.anchors_stride) for i in range(anchors.shape[0])]
    )
    self.anchors = anchors

  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)
    max_step = (self.get_num_samples() * self.config.epochs // batch_size)
    tf.constant(max_step, name="max_step")

  def parse_fn(self, file_name, classes, boxes, is_crowd):
    """Parse a single input sample
    """
    image = tf.read_file(file_name)
    image = tf.image.decode_png(image, channels=3)
    image = tf.to_float(image)

    if self.augmenter:
      is_training = (self.config.mode == "train")
      image, classes, boxes, is_crowd = \
        self.augmenter.augment(
          image,
          classes,
          boxes,
          is_crowd,
          self.config.resolution,
          is_training=is_training,
          speed_mode=False)

    return (image, classes, boxes, is_crowd)


  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)

    dataset = tf.data.Dataset.from_generator(
      generator=lambda: self.get_samples_fn(),
      output_types=(tf.string,
                    tf.int64,
                    tf.float32,
                    tf.int64))

    dataset = dataset.map(
      lambda file_name, classes, boxes, is_crowd: self.parse_fn(file_name, classes, boxes, is_crowd),
      num_parallel_calls=12)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(2)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

  def draw_boxes(self, im, labels, boxes, is_crowd):
      """
      Args:
          im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
          boxes (np.ndarray or list[BoxBase]): If an ndarray,
              must be of shape Nx4 where the second dimension is [x1, y1, x2, y2].
          labels: (list[str] or None)
          color: a 3-tuple (in range [0, 255]). By default will choose automatically.
      Returns:
          np.ndarray: a new image.
      """
      FONT = cv2.FONT_HERSHEY_SIMPLEX
      FONT_SCALE = 0.7
      if isinstance(boxes, list):
          arr = np.zeros((len(boxes), 4), dtype='int32')
          for idx, b in enumerate(boxes):
              assert isinstance(b, BoxBase), b
              arr[idx, :] = [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]
          boxes = arr
      else:
          boxes = boxes.astype('int32')
      if labels is not None:
          assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))

      im = im.copy()
      # COLOR = (218, 218, 218) if color is None else color
      COLOR = (255, 255, 55)

      areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
      sorted_inds = np.argsort(-areas)    # draw large ones first
      for i in sorted_inds:
          box = boxes[i, :]

          best_color = COLOR

          if labels is not None:
              label = self.cat_names[labels[i] - 1]
              # find the best placement for the text
              ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
              top_left = [box[0] + 1, box[1] - 1.3 * lineh]
              if top_left[1] < 0:     # out of image
                  top_left[1] = box[3] - 1.3 * lineh
              cv2.putText(im, label, (int(top_left[0]), int(top_left[1] + lineh)),
                          FONT, FONT_SCALE, color=best_color, lineType=cv2.LINE_AA)

          cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                        color=best_color, thickness=2)
      return im

  def draw_annotation(self, img, labels, boxes, is_crowd):
      """Will not modify img"""
      img = img / 255.0
      img = self.draw_boxes(img, labels, boxes, is_crowd)
      plt.imshow(img)
      plt.show()

def build(config, augmenter):
  return ObjectDetectionMSCOCOInputter(config, augmenter)
