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
from source.augmenter.external import vgg_preprocessing


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

    # Has to be more than num_gpu * batch_size_per_gpu
    # Otherwise no valid batch will be produced    
    self.TRAIN_NUM_SAMPLES = 117266 # train2014 + valminusminival2014
    self.EVAL_NUM_SAMPLES = 4952 # val2017 (same as test-dev2015)

    if self.config.mode == "infer":
      self.test_samples = self.config.test_samples
    elif self.config.mode == "export":
      pass
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
      elif self.config.mode == "export":
        self.num_samples = 1        
      elif self.config.mode == "eval":
        self.num_samples = self.EVAL_NUM_SAMPLES
      elif self.config.mode == "train":
        self.num_samples = self.TRAIN_NUM_SAMPLES
    return self.num_samples

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

    return ([image_id], image, classes, boxes, scale, translation, [file_name])

  def input_fn(self, test_samples=[]):
    if self.config.mode == "export":
      image = tf.placeholder(tf.float32,
                             shape=(self.config.resolution,
                                    self.config.resolution, 3),
                             name="input_image")
      image = tf.to_float(image)
      image = vgg_preprocessing._mean_image_subtraction(image)
      image = tf.expand_dims(image, 0)
      return image
    else:    
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

      dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None], [None, None, 3], [None], [None, 4], [None], [None], [None]))

      dataset = dataset.prefetch(2)

      iterator = dataset.make_one_shot_iterator()
      return iterator.get_next()


def build(config, augmenter):
  return ObjectDetectionMSCOCOInputter(config, augmenter)
