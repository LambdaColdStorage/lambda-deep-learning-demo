"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from modeler import Modeler


class ObjectDetectionModeler(Modeler):
  def __init__(self, args, net):
    super(ObjectDetectionModeler, self).__init__(args, net)
    

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.anchors = inputter.get_anchors()
    self.anchors_map = inputter.get_anchors_map()

  def create_nonreplicated_fn(self):
    pass

  def create_graph_fn(self, input):
    # forward graph 

    feat_labels = tf.zeros([self.anchors_map.shape[0], self.config.num_classes])
    feat_bboxes = tf.zeros([self.anchors_map.shape[0], 4])
    return feat_labels, feat_bboxes

  def create_eval_metrics_fn(self, predictions, labels):
    return accuracies

  def create_loss_fn(self, feat_labels, feat_bboxes, gt_classes, gt_bboxes):
    loss_classes = tf.zeros([1])

    loss_bboxes = tf.zeros([1])

    loss = loss_classes + loss_bboxes

    return loss 

  def create_grad_fn(self, loss, clipping=None):
    grads = tf.zeros([1024, 1024])
    return grads

  def create_detect_fn(feat_classes, feat_bboxes):
    detection_classes = tf.zeros([10, 1])
    detection_bboxes = tf.zeros([10, 4])
    return 

  def model_fn(self, x):
    images = x[0]
    classes = x[1]
    boxes = x[2]
    gt_classes = x[3]
    gt_bboxes = x[4]
    gt_mask = x[5]

    feat_classes, feat_bboxes = self.create_graph_fn(images)

    if self.config.mode == "train":
      loss = self.create_loss_fn(
        feat_classes,
        feat_bboxes,
        gt_classes,
        gt_bboxes)
      return loss
      # grads = self.create_grad_fn(loss)
      # return {"loss": loss,
      #         "grads": grads}
    elif self.config.mode == "eval":
      accuracies = self.create_eval_metrics_fn(
        feat_classes,
        feat_bboxes,
        classes,
        boxes)
      return {"accuracies": accuracies}
    elif self.config.mode == "infer":
      detection_classes, detection_bboxes = self.create_detect_fn(
        feat_classes,
        feat_bboxes)
      return {"classes": detection_classes,
              "bboxes": detection_bboxes}

def build(args, network):
  return ObjectDetectionModeler(args, network)
