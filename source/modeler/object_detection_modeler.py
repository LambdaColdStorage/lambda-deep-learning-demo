"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib

import tensorflow as tf

from modeler import Modeler


class ObjectDetectionModeler(Modeler):
  def __init__(self, args, net, loss):
    super(ObjectDetectionModeler, self).__init__(args, net)
    self.loss = loss
    self.feature_net = getattr(
      importlib.import_module("source.network." + self.config.feature_net),
      "net")
    self.feature_net_init_flag = True

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.anchors = inputter.get_anchors()
    self.anchors_map = inputter.get_anchors_map()

  def create_nonreplicated_fn(self):
    pass

  def create_graph_fn(self, inputs):
    # Inputs:
    # inputs: batch_size x h x w x 3
    # Outputs:
    # feat_classes: batch_size x num_anchors x num_classes
    # feat_bboxes: batch_size x num_anchors x 4

    # Feature net
    inputs, self.feature_net_init_flag = self.feature_net(
      inputs, self.config.data_format,
      is_training=False, init_flag=self.feature_net_init_flag,
      ckpt_path=self.config.feature_net_path)

    is_training = (self.config.mode == "train")
    return self.net(inputs, self.config.num_classes,
                    is_training=is_training, data_format=self.config.data_format)

  def create_eval_metrics_fn(self, predictions, labels):
    return accuracies

  def create_loss_fn(self, inputs, outputs):
    return self.loss(inputs, outputs) 

  def create_grad_fn(self, loss, clipping=None):
    grads = tf.zeros([1024, 1024])
    return grads

  def create_detect_fn(feat_classes, feat_bboxes):
    detection_classes = tf.zeros([10, 1])
    detection_bboxes = tf.zeros([10, 4])
    return 

  def model_fn(self, inputs):
    outputs = self.create_graph_fn(inputs[0])

    if self.config.mode == "train":
      loss = self.create_loss_fn(inputs, outputs)
      return loss

def build(args, network, loss):
  return ObjectDetectionModeler(args, network, loss)
