"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib
import math
import numpy as np

import tensorflow as tf

from .modeler import Modeler
from source.network.detection import detection_common


class ObjectDetectionModeler(Modeler):
  def __init__(self, args, net):
    super(ObjectDetectionModeler, self).__init__(args, net)
    self.loss = net.loss
    self.feature_net = getattr(
      importlib.import_module("source.network." + self.config.feature_net),
      "net")

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
    self.TRAIN_FG_IOU = 0.5
    self.TRAIN_BG_IOU = 0.5

    self.feature_net_init_flag = True

    self.config.CLASS_WEIGHTS = 1.0
    self.config.BBOXES_WEIGHTS = 1.0
    self.config.L2_REGULARIZATION = 0.00025

    # For get high mAP in evaluation.
    # For detection one should actually use higher RESULT_SCORE_THRESH (e.g. RESULT_SCORE_THRESH = 0.6)
    self.config.RESULT_SCORE_THRESH = 0.01
    self.config.RESULTS_PER_IM = 200
    self.config.NMS_THRESH = 0.45

    self.config.BACKBONE_OUTPUT_LAYER = "vgg_16/mod_pool5"
    self.config.FEATURE_LAYERS = ("vgg_16/conv4/conv4_3",
                                  "ssd_conv7", "ssd_conv8_2",
                                  "ssd_conv9_2", "ssd_conv10_2",
                                  "ssd_conv11_2", "ssd_conv12_2")
    self.config.FEATURE_MAP_SIZE = (64, 32, 16, 8, 4, 2, 1)

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    if self.config.mode == "train":
      self.learning_rate = self.create_learning_rate_fn(self.global_step)

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

  def create_graph_fn(self, inputs):
    # Args:
    #     inputs: batch_size x h x w x 3
    # Returns:
    #     feat_classes: batch_size x num_anchors x num_classes
    #     feat_bboxes: batch_size x num_anchors x 4

    # Feature net
    outputs = self.feature_net(
      inputs, self.config.data_format)

    is_training = (self.config.mode == "train")
    return self.net(outputs,
                    self.config.num_classes, self.num_anchors,
                    is_training=is_training, data_format=self.config.data_format)

  def create_eval_metrics_fn(self, predictions, labels):
    return accuracies

  def create_loss_fn(self, inputs, outputs):
    self.gether_train_vars()

    return self.loss(inputs, outputs, self.config.CLASS_WEIGHTS, self.config.BBOXES_WEIGHTS)

  def create_detect_fn(self, feat_classes, feat_bboxes):
    # Args:
    #     feat_classes:  batch_size x (num_anchors, num_classes)
    #     feat_bboxes:   batch_size x (num_anchors, 4)
    # Returns:
    #     detection_topk_scores: batch_size x (num_detections,), list of arrays
    #     detection_topk_bboxes: batch_size x (num_detections, 4), list of arrays

    score_classes = tf.nn.softmax(feat_classes)

    feat_bboxes = detection_common.decode_bboxes_batch(feat_bboxes, self.anchors_map)


    detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors = detection_common.detect_batch(
      score_classes, feat_bboxes, self.config, self.anchors_map)

    return detection_topk_scores, detection_topk_labels, detection_topk_bboxes,detection_topk_anchors


  def encode_gt(self, labels, boxes):


    def compute_gt(l, b):
      # Input:
      #     classes: num_obj
      #     boxes: num_obj x 4
      # Output:
      #     gt_labels: num_anchors
      #     gt_bboxes: num_anchors x 4
      #     gt_mask: num_anchors

      # Check there is at least one object in the image
      assert len(b) > 0

      # Compute IoU between anchors and boxes
      ret_iou = detection_common.np_iou(self.anchors_map, b)

      # Create mask:
      # foreground = 1
      # background = -1
      # neutral = 0

      # Forward selection
      max_idx = np.argmax(ret_iou, axis=1)
      max_iou = ret_iou[np.arange(ret_iou.shape[0]), max_idx]
      gt_labels = l[max_idx]
      gt_bboxes = b[max_idx, :]
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
      gt_labels[max_idx_reverse] = l
      gt_bboxes[max_idx_reverse] = b
      gt_mask[max_idx_reverse] = 1

      return gt_labels, gt_bboxes, gt_mask

    def encode(label, box):
      # encode list of objects into target gt
      ids = tf.reshape(tf.where(label > 0), [-1])
      label = tf.gather(label, ids)    
      box = tf.gather(box, ids)  

      gt_labels, gt_bboxes, gt_masks = tf.py_func(
        compute_gt, [label, box], (tf.int64, tf.float32, tf.int32))
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

      return gt_labels, gt_bboxes, gt_masks 


    list_labels = tf.unstack(labels, num=self.config.batch_size_per_gpu)
    list_boxes = tf.unstack(boxes, num=self.config.batch_size_per_gpu)

    gt_labels = []
    gt_bboxes = []
    gt_masks = []

    for l, b in zip(list_labels, list_boxes):
      gt_label, gt_bbox, gt_mask = encode(l, b)
      gt_labels.append(gt_label)
      gt_bboxes.append(gt_bbox)
      gt_masks.append(gt_mask)

    gt_labels = tf.stack(gt_labels)
    gt_bboxes = tf.stack(gt_bboxes)
    gt_masks = tf.stack(gt_masks)

    return gt_labels, gt_bboxes, gt_masks

  def model_fn(self, inputs):
    # Args:
    #     image_id = inputs[0]
    #     gt_image = inputs[1]
    #     gt_label = inputs[2]
    #     gt_boxes = decode_bboxes(inputs[3][0], self.anchors_map)
    #     gt_mask = inputs[4][0]
    #     scale = inputs[5]
    #     translation = inputs[6]
    #     file_name = inputs[7]
    # Returns:
    #     bboxes: x1, y1, x2, y2

    # Encode ground truth
    if self.config.mode == "train":
      self.anchors_map, self.num_anchors = self.get_anchors()

      image_id, image, labels, boxes, scale, translation, file_name = inputs

      gt_labels, gt_bboxes, gt_masks = self.encode_gt(labels, boxes)

      outputs = self.create_graph_fn(image)

      class_losses, bboxes_losses = self.create_loss_fn((image_id,
                                                         image,
                                                         gt_labels,
                                                         gt_bboxes,
                                                         gt_masks,
                                                         boxes,
                                                         scale,
                                                         translation),
                                                        outputs)

      loss_l2 = self.config.L2_REGULARIZATION * self.l2_regularization()
      
      loss = tf.identity(class_losses + bboxes_losses + loss_l2, "total_loss")

      grads = self.create_grad_fn(loss)
      return {"loss": loss,
              "class_losses": class_losses,
              "bboxes_losses": bboxes_losses,
              "grads": grads,
              "learning_rate": self.learning_rate,
              "gt_bboxes": gt_bboxes}


    # outputs = self.create_graph_fn(inputs[1])

    # if self.config.mode == "train":
    #   class_losses, bboxes_losses = self.create_loss_fn(inputs, outputs)

    #   loss_l2 = self.config.L2_REGULARIZATION * self.l2_regularization()
      
    #   loss = tf.identity(class_losses + bboxes_losses + loss_l2, "total_loss")

    #   grads = self.create_grad_fn(loss)
    #   return {"loss": loss,
    #           "class_losses": class_losses,
    #           "bboxes_losses": bboxes_losses,
    #           "grads": grads,
    #           "learning_rate": self.learning_rate,
    #           "gt_bboxes": inputs[3]}
    # elif self.config.mode == "infer":
    #   feat_classes = outputs[0]
    #   feat_bboxes = outputs[1]
    #   detection_scores, detection_labels, detection_bboxes, detection_anchors = self.create_detect_fn(feat_classes, feat_bboxes)

    #   return {"scores": detection_scores,
    #           "labels": detection_labels,
    #           "bboxes": detection_bboxes,
    #           "anchors": detection_anchors,
    #           "gt_bboxes": inputs[3],
    #           "gt_labels": inputs[2],
    #           "images": inputs[1],
    #           "predict_scores": feat_classes,
    #           "scales": inputs[5],
    #           "translations": inputs[6],
    #           "file_name": inputs[7]}
    # elif self.config.mode == "eval":
    #   feat_classes = outputs[0]
    #   feat_bboxes = outputs[1]
    #   detection_scores, detection_labels, detection_bboxes, detection_anchors = self.create_detect_fn(feat_classes, feat_bboxes)

    #   # make image_id a list so it conforms with detection results (also in form of list)
    #   return {"image_id": tf.unstack(inputs[0], self.config.batch_size_per_gpu),
    #           "scores": detection_scores,
    #           "labels": detection_labels,
    #           "bboxes": detection_bboxes,
    #           "scales": tf.unstack(inputs[5], self.config.batch_size_per_gpu),
    #           "translations": tf.unstack(inputs[6], self.config.batch_size_per_gpu),
    #           "file_name": tf.unstack(inputs[7], self.config.batch_size_per_gpu)}


def build(args, network):
  return ObjectDetectionModeler(args, network)
