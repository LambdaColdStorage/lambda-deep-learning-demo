"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib

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
    self.anchors, self.anchors_map, self.num_anchors = inputter.get_anchors()

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

    outputs = self.create_graph_fn(inputs[1])

    if self.config.mode == "train":
      class_losses, bboxes_losses = self.create_loss_fn(inputs, outputs)

      loss_l2 = self.config.L2_REGULARIZATION * self.l2_regularization()
      
      loss = tf.identity(class_losses + bboxes_losses + loss_l2, "total_loss")

      grads = self.create_grad_fn(loss)
      return {"loss": loss,
              "class_losses": class_losses,
              "bboxes_losses": bboxes_losses,
              "grads": grads,
              "learning_rate": self.learning_rate,
              "gt_bboxes": inputs[3]}
    elif self.config.mode == "infer":
      feat_classes = outputs[0]
      feat_bboxes = outputs[1]
      detection_scores, detection_labels, detection_bboxes, detection_anchors = self.create_detect_fn(feat_classes, feat_bboxes)

      return {"scores": detection_scores,
              "labels": detection_labels,
              "bboxes": detection_bboxes,
              "anchors": detection_anchors,
              "gt_bboxes": inputs[3],
              "gt_labels": inputs[2],
              "images": inputs[1],
              "predict_scores": feat_classes,
              "scales": inputs[5],
              "translations": inputs[6],
              "file_name": inputs[7]}
    elif self.config.mode == "eval":
      feat_classes = outputs[0]
      feat_bboxes = outputs[1]
      detection_scores, detection_labels, detection_bboxes, detection_anchors = self.create_detect_fn(feat_classes, feat_bboxes)

      # make image_id a list so it conforms with detection results (also in form of list)
      return {"image_id": tf.unstack(inputs[0], self.config.batch_size_per_gpu),
              "scores": detection_scores,
              "labels": detection_labels,
              "bboxes": detection_bboxes,
              "scales": tf.unstack(inputs[5], self.config.batch_size_per_gpu),
              "translations": tf.unstack(inputs[6], self.config.batch_size_per_gpu),
              "file_name": tf.unstack(inputs[7], self.config.batch_size_per_gpu)}


def build(args, network):
  return ObjectDetectionModeler(args, network)
