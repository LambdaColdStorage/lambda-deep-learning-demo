import os
import importlib
import math
import numpy as np

import tensorflow as tf

from source.network.detection import ssd_common

NAME_FEATURE_NET = "vgg_16_reduced"

CLASS_WEIGHTS = 1.0
BBOXES_WEIGHTS = 1.0

# Priorboxes
ANCHORS_STRIDE = [8, 16, 32, 64, 100, 300]
ANCHORS_ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# control the size of the default square priorboxes
# REF: https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp#L164
MIN_SIZE_RATIO = 15
MAX_SIZE_RATIO = 90
INPUT_DIM = 300

ANCHORS_MAP, NUM_ANCHORS = ssd_common.get_anchors(ANCHORS_STRIDE,
                                                  ANCHORS_ASPECT_RATIOS,
                                                  MIN_SIZE_RATIO,
                                                  MAX_SIZE_RATIO,
                                                  INPUT_DIM)
VGG_PARAMS_FILE = os.path.join(os.path.expanduser("~"), "demo/model/VGG_16_reduce/VGG_16_reduce.p")


def encode_gt(inputs, batch_size):
  image_id, image, labels, boxes, scale, translation, file_name = inputs
  gt_labels, gt_bboxes, gt_masks = ssd_common.encode_gt(labels, boxes, ANCHORS_MAP, batch_size)
  return gt_labels, gt_bboxes, gt_masks


def ssd_feature(outputs, data_format):
    outputs_conv6_2 = ssd_common.ssd_block(outputs, "conv6", data_format, [1, 2], [1, 3], [256, 512], ["SAME", "SAME"])
    outputs_conv7_2 = ssd_common.ssd_block(outputs_conv6_2, "conv7", data_format, [1, 2], [1, 3], [128, 256], ["SAME", "SAME"])
    outputs_conv8_2 = ssd_common.ssd_block(outputs_conv7_2, "conv8", data_format, [1, 2], [1, 3], [128, 256], ["SAME", "SAME"])
    outputs_conv9_2 = ssd_common.ssd_block(outputs_conv8_2, "conv9", data_format, [1, 2], [1, 3], [128, 256], ["SAME", "VALID"])
    return outputs_conv6_2, outputs_conv7_2, outputs_conv8_2, outputs_conv9_2


def net(inputs,
        num_classes,
        is_training, 
        data_format="channels_last"):

  image_id, image, labels, boxes, scale, translation, file_name = inputs
  
  feature_net = getattr(
    importlib.import_module("source.network." + NAME_FEATURE_NET),
    "net")

  outputs = feature_net(image, data_format, VGG_PARAMS_FILE)


  with tf.variable_scope(name_or_scope='SSD',
                         values=[outputs],
                         reuse=tf.AUTO_REUSE):

    outputs_conv4_3 = outputs[0]
    outputs_fc7 = outputs[1]

    # # Add shared features
    outputs_conv6_2, outputs_conv7_2, outputs_conv8_2, outputs_conv9_2 = ssd_feature(outputs_fc7, data_format)

    classes = []
    bboxes = []
    feature_layers = (outputs_conv4_3, outputs_fc7, outputs_conv6_2, outputs_conv7_2, outputs_conv8_2, outputs_conv9_2)
    name_layers = ("VGG/conv4_3", "VGG/fc7", "SSD/conv6_2", "SSD/conv7_2", "SSD/conv8_2", "SSD/conv9_2")

    for name, feat, num in zip(name_layers, feature_layers, NUM_ANCHORS):      
      # According to the original SSD paper, normalize conv4_3 with learnable scale
      # In pratice doing so indeed reduce the classification loss significantly
      if name == "VGG/conv4_3":
        l2_w_init = tf.constant_initializer([20.] * 512)
        weight_scale = tf.get_variable('l2_norm_scaler',
                                       initializer=[20.] * 512,
                                       trainable=is_training)        
        feat = tf.multiply(weight_scale,
                           tf.math.l2_normalize(feat, axis=-1, epsilon=1e-12))

      classes.append(ssd_common.class_graph_fn(feat, num_classes, num, name))
      bboxes.append(ssd_common.bbox_graph_fn(feat, num, name))

    classes = tf.concat(classes, axis=1)
    bboxes = tf.concat(bboxes, axis=1)

    return classes, bboxes


def loss(gt, outputs):
  return ssd_common.loss(gt, outputs, CLASS_WEIGHTS, BBOXES_WEIGHTS)


def detect(feat_classes, feat_bboxes, batch_size, num_classes):
  score_classes = tf.nn.softmax(feat_classes)

  feat_bboxes = ssd_common.decode_bboxes_batch(feat_bboxes, ANCHORS_MAP, batch_size)

  detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors = ssd_common.detect_batch(
    score_classes, feat_bboxes, ANCHORS_MAP, batch_size, num_classes)

  return detection_topk_scores, detection_topk_labels, detection_topk_bboxes,detection_topk_anchors
