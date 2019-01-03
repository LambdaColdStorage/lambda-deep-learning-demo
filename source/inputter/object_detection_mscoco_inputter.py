"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function
import os
import numpy as np
import math

import tensorflow as tf

from .inputter import Inputter
from pycocotools.coco import COCO
from source.network.detection import detection_common


JSON_TO_IMAGE = {
    "train2017": "train2017",
    "val2017": "val2017",
    "train2014": "train2014",
    "val2014": "val2014",
    "valminusminival2014": "val2014",
    "minival2014": "val2014",
    "test2014": "test2014",
    "test-dev2015": "val2017"
}


class ObjectDetectionMSCOCOInputter(Inputter):
  def __init__(self, config, augmenter):
    super(ObjectDetectionMSCOCOInputter, self).__init__(config, augmenter)

    self.category_id_to_class_id = None
    self.class_id_to_category_id = None
    self.cat_names = None

    # self.priorbox_path = os.path.join(os.path.expanduser("~"), "git/caffe_ssd/SSD_512x512_priorbox.p")
    self.priorvariance = [0.1, 0.1, 0.2, 0.2]

    self.num_anchors = []
    self.anchors_map = None
    self.anchors_stride = [8, 16, 32, 64, 128, 256, 512]
    self.anchors_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    # control the size of the default square priorboxes
    # REF: https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp#L164
    self.min_ratio = 10
    self.max_ratio = 90
    self.min_dim = 512
    
    # Has to be more than num_gpu * batch_size_per_gpu
    # Otherwise no valid batch will be produced
    # self.TRAIN_NUM_SAMPLES = 2048
    # self.EVAL_NUM_SAMPLES = 2048

    # self.TRAIN_NUM_SAMPLES = 82784
    # self.EVAL_NUM_SAMPLES = 2048

    # self.TRAIN_NUM_SAMPLES = 117266 # train2017
    # self.EVAL_NUM_SAMPLES = 2048

    # self.TRAIN_NUM_SAMPLES = 82081 # train2014
    # self.TRAIN_NUM_SAMPLES = 35185 # valminusminival2014
    
    self.TRAIN_NUM_SAMPLES = 117266 # train2014 + valminusminival2014
    self.EVAL_NUM_SAMPLES = 4952 # val2017 (same as test-dev2015)

    self.TRAIN_FG_IOU = 0.5
    self.TRAIN_BG_IOU = 0.5

    if self.config.mode == "infer":
      self.test_samples = self.config.test_samples
    else:
      self.parse_coco()

    self.num_samples = self.get_num_samples()


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

      if self.config.mode == "train":
        for img in imgs:
          self.parse_gt(coco, self.category_id_to_class_id, img)

      samples.extend(imgs)

    # Filter out images that has no object.
    if self.config.mode == "train":
      samples = list(filter(
        lambda sample: len(
          sample['boxes'][sample['is_crowd'] == 0]) > 0, samples))

    self.samples = samples

  def get_num_samples(self):
    if not hasattr(self, 'num_samples'):
      if self.config.mode == "infer":
        self.num_samples = len(self.test_samples)
      elif self.config.mode == "eval":
        self.num_samples = self.EVAL_NUM_SAMPLES
      elif self.config.mode == "train":
        self.num_samples = self.TRAIN_NUM_SAMPLES
    return self.num_samples


  def get_anchors(self):

    if self.anchors_map is None:

      step = int(math.floor((self.max_ratio - self.min_ratio) / (len(self.anchors_aspect_ratios) - 2)))
      min_sizes = []
      max_sizes = []
      for ratio in xrange(self.min_ratio, self.max_ratio + 1, step):
        min_sizes.append(self.min_dim * ratio / 100.)
        max_sizes.append(self.min_dim * (ratio + step) / 100.)
      min_sizes = [self.min_dim * 4 / 100.] + min_sizes
      max_sizes = [self.min_dim * 10 / 100.] + max_sizes
      min_sizes = [math.floor(x) for x in min_sizes]
      max_sizes = [math.floor(x) for x in max_sizes]

      list_priorbox, list_num_anchors = detection_common.ssd_create_priorbox(
        self.min_dim,
        self.anchors_aspect_ratios,
        self.anchors_stride,
        min_sizes,
        max_sizes)
      self.anchors_map = np.concatenate(list_priorbox, axis=0)
      self.num_anchors = list_num_anchors

      return self.anchors_map, self.num_anchors


  def get_samples_fn(self):
    # Args:
    # Returns:
    #     sample["id"]: int64, image id
    #     sample["file_name"]: , string, path to image
    #     sample["class"]: (...,), int64
    #     sample["boxes"]: (..., 4), float32
    # Read image
    if self.config.mode == "infer":
      for file_name in self.test_samples:
        yield (0,
               file_name,
               np.empty([1], dtype=np.int32),
               np.empty([1, 4]))
    elif self.config.mode == "eval":
      for sample in self.samples[0:self.num_samples]:
        yield(sample["id"],
              sample["file_name"],
              np.empty([1], dtype=np.int32),
              np.empty([1, 4]))
    else:
      for sample in self.samples[0:self.num_samples]:
        # remove crowd objects
        mask = sample['is_crowd'] == 0
        sample["class"] = sample["class"][mask]
        sample["boxes"] = sample["boxes"][mask, :]
        sample["is_crowd"] = sample["is_crowd"][mask]

        yield (sample["id"],
               sample["file_name"],
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
        # normalize box to [0, 1]
        obj['bbox'] = [x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)]
        valid_objs.append(obj)

    boxes = np.asarray([obj['bbox'] for obj in valid_objs], dtype='float32')  # (n, 4)

    cls = np.asarray([
        category_id_to_class_id[obj['category_id']]
        for obj in valid_objs], dtype='int32')  # (n,)

    is_crowd = np.asarray([obj['iscrowd'] for obj in valid_objs], dtype='int8')

    img['boxes'] = boxes # nx4
    img['class'] = cls # n, always >0
    img['is_crowd'] = is_crowd # n,

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
    gt_labels[max_idx_reverse] = classes
    gt_bboxes[max_idx_reverse] = boxes
    gt_mask[max_idx_reverse] = 1

    return gt_labels, gt_bboxes, gt_mask

  def parse_fn(self, image_id, file_name, classes, boxes):
    """Parse a single input sample
    """
    image = tf.read_file(file_name)
    image = tf.image.decode_png(image, channels=3)
    image = tf.to_float(image)

    scale = [0, 0]
    translation = [0, 0]
    if self.augmenter:
      is_training = (self.config.mode == "train")
      image, classes, boxes, scale, translation = self.augmenter.augment(
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
    elif self.config.mode == "eval":
      # For object detection use external library for evaluation
      # Skip gt here
      gt_labels = tf.zeros([1], dtype=tf.int64)
      gt_bboxes = tf.zeros([1, 4], dtype=tf.float32)
      gt_mask = tf.zeros([1], dtype=tf.int32)
    elif self.config.mode == "train":
      gt_labels, gt_bboxes, gt_mask = tf.py_func(
        self.compute_gt, [classes, boxes], (tf.int64, tf.float32, tf.int32))
      # Encode the shift between gt_bboxes and anchors_map
      gt_bboxes = detection_common.encode_bbox_target(
        gt_bboxes, self.anchors_map)

      # scale with variance 
      cx, cy, w, h = tf.unstack(gt_bboxes, 4, axis=1)
      cx = tf.scalar_mul(1.0 / self.priorvariance[0], cx)
      cy = tf.scalar_mul(1.0 / self.priorvariance[1], cy)
      w = tf.scalar_mul(1.0 / self.priorvariance[2], w)
      h = tf.scalar_mul(1.0 / self.priorvariance[3], h)
      gt_bboxes = tf.concat([tf.expand_dims(cx, -1),
                             tf.expand_dims(cy, -1),
                             tf.expand_dims(w, -1),
                             tf.expand_dims(h, -1)], axis=1)

    return (image_id, image, gt_labels, gt_bboxes, gt_mask, scale, translation, file_name)

  def input_fn(self, test_samples=[]):
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)

    dataset = tf.data.Dataset.from_generator(
      generator=lambda: self.get_samples_fn(),
      output_types=(tf.int64,
                    tf.string,
                    tf.int64,
                    tf.float32))

    if self.config.mode == "train":
      dataset = dataset.shuffle(self.get_num_samples())

    dataset = dataset.repeat(self.config.epochs)

    dataset = dataset.map(
      lambda image_id, file_name, classes, boxes: self.parse_fn(
        image_id, file_name, classes, boxes),
      num_parallel_calls=12)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(2)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def build(config, augmenter):
  return ObjectDetectionMSCOCOInputter(config, augmenter)
