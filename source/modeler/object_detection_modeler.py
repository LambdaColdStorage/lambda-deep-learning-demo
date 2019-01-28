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

class ObjectDetectionModeler(Modeler):
  def __init__(self, args, net):
    super(ObjectDetectionModeler, self).__init__(args, net)
    self.loss = net.loss
    self.encode_gt = net.encode_gt
    self.detect = net.detect

    # self.config.L2_REGULARIZATION = 0.00025

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    

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

    # # Feature net
    # outputs = self.feature_net(
    #   inputs, self.config.data_format)

    is_training = (self.config.mode == "train")
    return self.net(inputs,
                    self.config.num_classes,
                    is_training=is_training,
                    feature_net=self.config.feature_net,
                    feature_net_path=self.config.feature_net_path,
                    data_format=self.config.data_format)


  def create_eval_metrics_fn(self, predictions, labels):
    return accuracies

  def create_loss_fn(self, gt, outputs):
    self.gether_train_vars()

    return self.loss(gt, outputs)

  def create_detect_fn(self, feat_classes, feat_bboxes):
    # Args:
    #     feat_classes:  batch_size x (num_anchors, num_classes)
    #     feat_bboxes:   batch_size x (num_anchors, 4)
    # Returns:
    #     detection_topk_scores: batch_size x (num_detections,), list of arrays
    #     detection_topk_bboxes: batch_size x (num_detections, 4), list of arrays

    return self.detect(feat_classes,
                       feat_bboxes,
                       self.config.batch_size_per_gpu,
                       self.config.num_classes,
                       self.config.confidence_threshold)

  def model_fn(self, inputs):

    outputs = self.create_graph_fn(inputs)

    if self.config.mode == 'train':
      gt = self.encode_gt(inputs,
                          self.config.batch_size_per_gpu,
                          )
      
      class_losses, bboxes_losses = self.create_loss_fn(gt, outputs)

      # loss_l2 = self.config.L2_REGULARIZATION * self.l2_regularization()

      loss_l2 = self.l2_regularization()

      loss = tf.identity(class_losses + bboxes_losses + loss_l2, "total_loss")

      grads = self.create_grad_fn(loss)

      return {"loss": loss,
              "class_losses": class_losses,
              "bboxes_losses": bboxes_losses,
              "grads": grads,
              "learning_rate": self.learning_rate,
              "gt_bboxes": gt[1]}
    elif self.config.mode == 'eval':
      feat_classes, feat_bboxes = outputs
      detection_scores, detection_labels, detection_bboxes, detection_anchors = self.create_detect_fn(feat_classes, feat_bboxes)
      # make image_id a list so it conforms with detection results (also in form of list)
      return {"image_id": tf.unstack(inputs[0], self.config.batch_size_per_gpu),
              "scores": detection_scores,
              "labels": detection_labels,
              "bboxes": detection_bboxes,
              "scales": tf.unstack(inputs[4], self.config.batch_size_per_gpu),
              "translations": tf.unstack(inputs[5], self.config.batch_size_per_gpu),
              "file_name": tf.unstack(inputs[6], self.config.batch_size_per_gpu)}
    elif self.config.mode == 'infer':
      feat_classes, feat_bboxes = outputs
      detection_scores, detection_labels, detection_bboxes, detection_anchors = self.create_detect_fn(feat_classes, feat_bboxes)      
      return {"scores": detection_scores,
              "labels": detection_labels,
              "bboxes": detection_bboxes,
              "anchors": detection_anchors,
              "images": inputs[1],
              "predict_scores": feat_classes,
              "scales": inputs[4],
              "translations": inputs[5],
              "file_name": inputs[6][0]}
    elif self.config.mode == "export":
      feat_classes, feat_bboxes = outputs
      detection_scores, detection_labels, detection_bboxes, detection_anchors = self.create_detect_fn(feat_classes, feat_bboxes)      
      output_scores = tf.identity(detection_scores, name="output_scores")
      output_labels = tf.identity(detection_labels, name="output_labels")
      output_bboxes = tf.identity(detection_bboxes, name="output_bboxes")
      return output_scores, output_labels, output_bboxes

def build(args, network):
  return ObjectDetectionModeler(args, network)
