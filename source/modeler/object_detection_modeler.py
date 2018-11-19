"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib

import tensorflow as tf

from modeler import Modeler


class ObjectDetectionModeler(Modeler):
  def __init__(self, args, net):
    super(ObjectDetectionModeler, self).__init__(args, net)

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

  def ssd_feature_fn(self, feat):
    data_format = 'channels_last'
    kernel_init = tf.variance_scaling_initializer()
    output = tf.layers.conv2d(inputs=feat,
                              filters=512,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              padding=('SAME'),
                              data_format=data_format,
                              kernel_initializer=kernel_init,
                              activation=tf.nn.relu,
                              name='feat_ssd')
    return output

  def class_graph_fn(self, feat):
    data_format = 'channels_last'
    kernel_init = tf.variance_scaling_initializer()
    output = tf.layers.conv2d(inputs=feat,
                              filters= 5 * 3 * self.config.num_classes,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              padding=('SAME'),
                              data_format=data_format,
                              kernel_initializer=kernel_init,
                              activation=None)
    output = tf.reshape(output,
                        [self.config.batch_size_per_gpu,
                         -1,
                         self.config.num_classes],
                        name='feat_classes')
    return output


  def bbox_graph_fn(self, feat):
    data_format = 'channels_last'
    kernel_init = tf.variance_scaling_initializer()
    output = tf.layers.conv2d(inputs=feat,
                              filters= 5 * 3 * 4,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              padding=('SAME'),
                              data_format=data_format,
                              kernel_initializer=kernel_init,
                              activation=None)
    output = tf.reshape(output,
                        [self.config.batch_size_per_gpu,
                         -1,
                         4],
                        name='feat_bboxes')    
    return output

  def create_graph_fn(self, inputs):
    # Inputs:
    # inputs: batch_size x h x w x 3
    # Outputs:
    # feat_classes: batch_size x num_anchors x num_classes
    # feat_bboxes: batch_size x num_anchors x 4

    # Feature net
    (logits, feat), self.feature_net_init_flag = self.feature_net(
      inputs, self.config.data_format,
      is_training=False, init_flag=self.feature_net_init_flag,
      ckpt_path=self.config.feature_net_path)

    # Shared SSD feature layer
    feat_ssd = self.ssd_feature_fn(logits)

    # Class head
    feat_classes = self.class_graph_fn(feat_ssd)

    # BBox head
    feat_bboxes = self.bbox_graph_fn(feat_ssd)

    return feat_classes, feat_bboxes

  def create_eval_metrics_fn(self, predictions, labels):
    return accuracies

  def create_loss_classes_fn(self, feat_classes, gt_classes, mask):
    logits = tf.boolean_mask(
      feat_classes,
      mask,
      axis=1)
    labels = tf.boolean_mask(
      gt_classes,
      mask,
      axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits,
      labels=labels)
    return loss

  def create_loss_bboxes_fn(self, feat_bboxes, gt_bboxes, mask):
    pred = tf.boolean_mask(
      feat_bboxes,
      mask,
      axis=1)
    gt = tf.boolean_mask(
      gt_bboxes,
      mask,
      axis=1)
    abs_diff = tf.abs(pred - gt)
    minx = tf.minimum(abs_diff, 1)
    loss = tf.reduce_sum(0.5 * ((abs_diff - 1) * minx + abs_diff))
    return loss

  def create_loss_fn(self, feat_classes, feat_bboxes, gt_classes, gt_bboxes, gt_mask):

    mask = tf.math.not_equal(gt_mask, 0)
    mask.set_shape([None])

    loss_classes = self.create_loss_classes_fn(feat_classes, gt_classes, mask)

    loss_bboxes = self.create_loss_bboxes_fn(feat_bboxes, gt_bboxes, mask)

    loss = loss_bboxes

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
        gt_bboxes,
        gt_mask)
      return loss
      # grads = self.create_grad_fn(loss)
      # return {"loss": loss,
      #         "grads": grads}
    # elif self.config.mode == "eval":
    #   accuracies = self.create_eval_metrics_fn(
    #     feat_classes,
    #     feat_bboxes,
    #     classes,
    #     boxes)
    #   return {"accuracies": accuracies}
    # elif self.config.mode == "infer":
    #   detection_classes, detection_bboxes = self.create_detect_fn(
    #     feat_classes,
    #     feat_bboxes)
    #   return {"classes": detection_classes,
    #           "bboxes": detection_bboxes}

def build(args, network):
  return ObjectDetectionModeler(args, network)
