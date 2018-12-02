"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib
import numpy as np

import tensorflow as tf

from modeler import Modeler
from source.network.detection import detection_common


class ObjectDetectionModeler(Modeler):
  def __init__(self, args, net, loss):
    super(ObjectDetectionModeler, self).__init__(args, net)
    self.loss = loss
    self.feature_net = getattr(
      importlib.import_module("source.network." + self.config.feature_net),
      "net")
    self.feature_net_init_flag = True

    self.config.CLASS_WEIGHTS = 1.0
    self.config.BBOXES_WEIGHTS = 50.0

    self.config.RESULT_SCORE_THRESH = 0.8
    self.config.RESULTS_PER_IM = 10
    self.config.NMS_THRESH = 0.5

    self.config.BACKBONE_OUTPUT_LAYER = "vgg_16/conv5/conv5_3"
    self.config.FEATURE_LAYERS = ("vgg_16/conv4/conv4_3",
                                  "ssd_conv7", "ssd_conv8_2",
                                  "ssd_conv9_2", "ssd_conv10_2",
                                  "ssd_conv11_2", "ssd_conv12_2")
    self.config.FEATURE_MAP_SIZE = (64, 32, 16, 8, 4, 2, 1)

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.anchors, self.anchors_map, self.num_anchors = inputter.get_anchors()
    # self.anchors = inputter.get_anchors()
    # self.anchors_map = inputter.get_anchors_map()

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    if self.config.mode == "train":
      self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_graph_fn(self, inputs):
    # Args:
    #     inputs: batch_size x h x w x 3
    # Returns:
    #     feat_classes: batch_size x num_anchors x num_classes
    #     feat_bboxes: batch_size x num_anchors x 4

    # Feature net
    inputs, self.feature_net_init_flag = self.feature_net(
      inputs, self.config.data_format,
      is_training=False, init_flag=self.feature_net_init_flag,
      ckpt_path=self.config.feature_net_path)

    is_training = (self.config.mode == "train")
    return self.net(inputs,
                    self.config.BACKBONE_OUTPUT_LAYER,
                    self.config.FEATURE_LAYERS,
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

  def model_fn(self, inputs):
    # gt_image = inputs[0]
    # gt_label = inputs[1]
    # gt_boxes = decode_bboxes(inputs[2][0], self.anchors_map)
    # gt_mask = inputs[3][0]
    # return gt_image, gt_label, gt_boxes, gt_mask

    outputs = self.create_graph_fn(inputs[0])

    if self.config.mode == "train":
      class_losses, bboxes_losses = self.create_loss_fn(inputs, outputs)
      loss = class_losses + bboxes_losses
      grads = self.create_grad_fn(loss)
      return {"loss": loss,
              "class_losses": class_losses,
              "bboxes_losses": bboxes_losses,
              "grads": grads,
              "learning_rate": self.learning_rate,
              "gt_bboxes": inputs[2]}
    elif self.config.mode == "infer":

      feat_classes = outputs[0]
      feat_bboxes = outputs[1]

      detection_scores, detection_labels, detection_bboxes,detection_anchors = self.create_detect_fn(feat_classes, feat_bboxes)

      return {"scores": detection_scores,
              "labels": detection_labels,
              "bboxes": detection_bboxes,
              "anchors": detection_anchors,
              "gt_bboxes": inputs[2],
              "gt_labels": inputs[1],
              "images": inputs[0],
              "predict_scores": feat_classes} 


def build(args, network, loss):
  return ObjectDetectionModeler(args, network, loss)
