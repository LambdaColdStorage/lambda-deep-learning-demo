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
import math

import tensorflow as tf

from inputter import Inputter
from pycocotools.coco import COCO
from source.network.detection import detection_common


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
    self.anchors_map = None

    self.TRAIN_NUM_SAMPLES = 32

    self.TRAIN_SAMPLES_PER_IMAGE = 256
    self.TRAIN_FG_IOU = 0.5
    self.TRAIN_BG_IOU = 0.5
    self.TRAIN_FG_RATIO = 0.5

    if self.config.mode == "infer":
      self.test_samples = self.config.test_samples
    else:
      self.parse_coco()

  def parse_coco(self):
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
    self.samples = samples   

  def get_num_samples(self):
    # return self.num_samples
    if self.num_samples < 0:
      if self.config.mode == "infer":
        self.num_samples = len(self.test_samples)
      else:
        # TODO: find a better way to define num_samples
        return self.TRAIN_NUM_SAMPLES
    return self.num_samples

  def get_anchors(self):
    if self.anchors is None:
      self.generate_anchors()
    return self.anchors

  def get_anchors_map(self):
    if self.anchors_map is None:
      self.generate_anchors_map()
    return self.anchors_map

  def get_samples_fn(self):
    # Args:
    # Returns:
    #     sample["file_name"]: , string, path to image
    #     sample["class"]: (...,), int32
    #     sample["boxes"]: (..., 4), float32
    # Read image
    if self.config.mode == "infer":
      for file_name in self.test_samples:
        yield (file_name,
               np.empty([1], dtype=np.int32),
               np.empty([1, 4]))
    else:
      for sample in self.samples[0:self.TRAIN_NUM_SAMPLES]:
        # remove crowd objects
        mask = sample['is_crowd'] == 0
        sample["class"] = sample["class"][mask]
        sample["boxes"] = sample["boxes"][mask, :]
        sample["is_crowd"] = sample["is_crowd"][mask]

        yield (sample["file_name"],
               sample["class"],
               sample["boxes"])

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
      [1, 1, self.anchors_stride, self.anchors_stride], dtype=np.float32) - 1
    anchors = self._ratio_enum(
      anchor, np.array(self.anchors_aspect_ratios, dtype=np.float32))
    anchors = np.vstack(
        [self._scale_enum(
         anchors[i, :],
         np.array(self.anchors_sizes, dtype=np.float32) / self.anchors_stride) for i in range(anchors.shape[0])]
    )
    self.anchors = anchors

  def generate_anchors_map(self):
    if self.anchors is None:
      self.generate_anchors()
    num_anchors = self.anchors.shape[0]

    map_resolution = int(np.ceil(
      self.anchors_stride * np.ceil(self.config.resolution / float(self.anchors_stride)) / float(self.anchors_stride)))
    shifts = np.arange(0, map_resolution) * self.anchors_stride

    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.ravel()
    shift_y = shift_y.ravel()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()

    A = self.anchors.shape[0]
    K = shifts.shape[0]
    self.anchors_map = (
        self.anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    )
    self.anchors_map = self.anchors_map.reshape((K * A, 4))
    self.anchors_map = np.float32(self.anchors_map)

  def create_nonreplicated_fn(self):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)

    max_step = (self.get_num_samples() * self.config.epochs // batch_size)

    tf.constant(max_step, name="max_step")

  def compute_gt(self, classes, boxes):
    # Input:
    #     classes: num_obj 
    #     boxes: num_obj x 4
    # Output:
    #     gt_labels: num_anchors
    #     gt_bboxes: num_anchors x 4
    #     gt_mask: num_anchors 

    # Check there is at least one object in the image
    assert len(boxes) > 0

    # Compute IoU between anchors and boxes
    ret_iou = detection_common.np_iou(self.anchors_map, boxes)

    # Create mask:
    # foreground = 1
    # background = -1
    # neutral = 0

    # Forward selection
    max_idx = np.argmax(ret_iou, axis=1)
    max_iou = ret_iou[np.arange(ret_iou.shape[0]), max_idx]    
    gt_labels = classes[max_idx]
    gt_bboxes = boxes[max_idx, :]    
    gt_mask = np.zeros(ret_iou.shape[0], dtype=np.int32)
    fg_idx = np.where(max_iou > self.TRAIN_FG_IOU)[0]
    bg_idx = np.where(max_iou < self.TRAIN_BG_IOU)[0]
    gt_mask[fg_idx] = 1
    gt_mask[bg_idx] = -1
    # Set the bg object to class 0
    gt_labels[bg_idx] = 0

    # Reverse selection
    # Make sure every gt object is matched to at least one anchor
    max_idx_reverse = np.argmax(ret_iou, axis=0)
    max_iou_reverse = ret_iou[max_idx_reverse, np.arange(ret_iou.shape[1])]
    gt_labels[max_idx_reverse] = classes
    gt_bboxes[max_idx_reverse] = boxes
    gt_mask[max_idx_reverse] = 1

    # Balance & Sub-sample fg and bg objects
    fg_ids = np.where(gt_mask == 1)[0]
    fg_extra = (len(fg_ids) -
                int(math.floor(self.TRAIN_SAMPLES_PER_IMAGE * self.TRAIN_FG_RATIO)))
    if fg_extra > 0:
      random_fg_ids = np.random.choice(fg_ids, fg_extra, replace=False)
      gt_mask[random_fg_ids] = 0

    bg_ids = np.where(gt_mask == -1)[0]
    bg_extra = len(bg_ids) - (self.TRAIN_SAMPLES_PER_IMAGE - sum(gt_mask == 1))
    if bg_extra > 0:
      random_bg_ids = np.random.choice(bg_ids, bg_extra, replace=False)
      gt_mask[random_bg_ids] = 0
    
    return gt_labels, gt_bboxes, gt_mask

  def parse_fn(self, file_name, classes, boxes):
    """Parse a single input sample
    """
    image = tf.read_file(file_name)
    image = tf.image.decode_png(image, channels=3)
    image = tf.to_float(image)

    if self.augmenter:
      is_training = (self.config.mode == "train")
      image, classes, boxes = \
        self.augmenter.augment(
          image,
          classes,
          boxes,
          self.config.resolution,
          is_training=is_training,
          speed_mode=False)

    if self.config.mode == "infer":
      gt_labels = tf.zeros([1], dtype=tf.int64)
      gt_bboxes = tf.zeros([1, 4], dtype=tf.float32)
      gt_mask = tf.zeros([1], dtype=tf.int32)
    else:
      gt_labels, gt_bboxes, gt_mask = tf.py_func(self.compute_gt,
                                        [classes, boxes],
                                        (tf.int64, tf.float32, tf.int32))

      # Encode the shift between gt_bboxes and anchors_map
      gt_bboxes = detection_common.encode_bbox_target(gt_bboxes, self.anchors_map)

    return (image, gt_labels, gt_bboxes, gt_mask)

  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)

    dataset = tf.data.Dataset.from_generator(
      generator=lambda: self.get_samples_fn(),
      output_types=(tf.string,
                    tf.int64,
                    tf.float32))

    if self.config.mode == "train":
      dataset = dataset.shuffle(self.get_num_samples())

    dataset = dataset.repeat(self.config.epochs)

    dataset = dataset.map(
      lambda file_name, classes, boxes: self.parse_fn(file_name, classes, boxes),
      num_parallel_calls=12)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(2)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

  def draw_boxes(self, im, labels, boxes):
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

  def draw_annotation(self, img, labels, boxes):
      """Will not modify img"""
      img = img / 255.0
      img = self.draw_boxes(img, labels, boxes)
      plt.imshow(img)
      plt.show()

def build(config, augmenter):
  return ObjectDetectionMSCOCOInputter(config, augmenter)
