import numpy as np
import math

import tensorflow as tf

TRAIN_SAMPLES_PER_IMAGE = 512
TRAIN_FG_RATIO = 0.5
KERNEL_INIT = tf.contrib.layers.xavier_initializer()


def ssd_block(outputs, name, data_format, conv_strides, filter_size, num_filters):

    num_conv = len(conv_strides)
    for i in range(num_conv):
        stride = conv_strides[i]
        w = filter_size[i]
        num_filter = num_filters[i]

        if stride == 2:
            # Use customized padding when stride == 2
            # https://stackoverflow.com/questions/42924324/tensorflows-asymmetric-padding-assumptions
            if data_format == "channels_last":
              outputs = tf.pad(outputs, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
            else:
              outputs = tf.pad(outputs, [[0, 0], [0, 0], [1, 0], [1, 0]], "CONSTANT")
            padding_strategy = "VALID"
        elif w == 4:
            # Use customized padding when for the last ssd feature layer (filter_size == 4)
            if data_format == "channels_last":
              outputs = tf.pad(outputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            else:
              outputs = tf.pad(outputs, [[0, 0], [0, 0], [1, 1], [1, 1]], "CONSTANT")
            padding_strategy = "VALID"
        else:
            padding_strategy = "SAME"
        outputs = tf.layers.conv2d(
                outputs,
          filters=num_filter,
          kernel_size=(w, w),
          strides=(stride, stride),
          padding=(padding_strategy),
          data_format=data_format,
          kernel_initializer=KERNEL_INIT,
          activation=tf.nn.relu,
          name=name + "_" + str(i + 1))
    return outputs

def ssd_feature(outputs, data_format):
    outputs_conv6_2 = ssd_block(outputs, "conv6", data_format, [1, 2], [1, 3], [128, 256])
    outputs_conv7_2 = ssd_block(outputs_conv6_2, "conv7", data_format, [1, 2], [1, 3], [128, 256])
    outputs_conv8_2 = ssd_block(outputs_conv7_2, "conv8", data_format, [1, 2], [1, 3], [128, 256])
    outputs_conv9_2 = ssd_block(outputs_conv8_2, "conv9", data_format, [1, 2], [1, 3], [128, 256])
    outputs_conv10_2 = ssd_block(outputs_conv9_2, "conv10", data_format, [1, 1], [1, 4], [128, 256])
    return outputs_conv6_2, outputs_conv7_2, outputs_conv8_2, outputs_conv9_2, outputs_conv10_2

def class_graph_fn(feat, num_classes, num_anchors, layer):
  data_format = 'channels_last'
  output = tf.layers.conv2d(inputs=feat,
                            filters=num_anchors * num_classes,
                            kernel_size=[3, 3],
                            strides=(1, 1),
                            padding=('SAME'),
                            data_format=data_format,
                            kernel_initializer=KERNEL_INIT,
                            activation=None,
                            name="class/" + layer)
  output = tf.reshape(output,
                      [tf.shape(output)[0],
                       -1,
                       num_classes],
                      name='feat_classes' + layer)
  return output


def bbox_graph_fn(feat, num_anchors, layer):
  data_format = 'channels_last'
  output = tf.layers.conv2d(inputs=feat,
                            filters=num_anchors * 4,
                            kernel_size=[3, 3],
                            strides=(1, 1),
                            padding=('SAME'),
                            data_format=data_format,
                            kernel_initializer=KERNEL_INIT,
                            activation=None,
                            name="bbox/" + layer)
  output = tf.reshape(output,
                      [tf.shape(output)[0],
                       -1,
                       4],
                      name='feat_bboxes' + layer)
  return output


def create_loss_classes_fn(logits_classes, gt_labels, fg_index, bg_index):

  fg_labels = tf.gather(gt_labels, fg_index)
  bg_labels = tf.gather(gt_labels, bg_index)

  fg_logits = tf.gather(logits_classes, fg_index)
  bg_logits = tf.gather(logits_classes, bg_index)

  fg_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
    logits=fg_logits,
    labels=fg_labels))
  bg_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
    logits=bg_logits,
    labels=bg_labels))  

  return fg_loss + bg_loss


def create_loss_bboxes_fn(logits_bboxes, gt_bboxes, fg_index):
  pred = tf.gather(logits_bboxes, fg_index)
  gt = tf.gather(gt_bboxes, fg_index)

  loss = tf.losses.huber_loss(gt, pred, delta=0.5)

  return loss


def net(outputs,
        num_classes, num_anchors,
        is_training, data_format="channels_last"):

  with tf.variable_scope(name_or_scope='SSD',
                         values=[outputs],
                         reuse=tf.AUTO_REUSE):

    outputs_conv4_3 = outputs[0]
    outputs_fc7 = outputs[1]

    # Add shared features
    outputs_conv6_2, outputs_conv7_2, outputs_conv8_2, outputs_conv9_2, outputs_conv10_2 = ssd_feature(outputs_fc7, data_format)

    classes = []
    bboxes = []
    feature_layers = (outputs_conv4_3, outputs_fc7, outputs_conv6_2, outputs_conv7_2, outputs_conv8_2, outputs_conv9_2, outputs_conv10_2)
    name_layers = ("VGG/conv4_3", "VGG/fc7", "SSD/conv6_2", "SSD/conv7_2", "SSD/conv8_2", "SSD/conv9_2", "SSD/conv10_2")

    for name, feat, num in zip(name_layers, feature_layers, num_anchors):      
      # According to the original SSD paper, normalize conv4_3 with learnable scale
      # In pratice doing so indeed reduce the classification loss significantly
      if name == "VGG/conv4_3":
        l2_w_init = tf.constant_initializer([20.] * 512)
        weight_scale = tf.get_variable('l2_norm_scaler',
                                       initializer=[20.] * 512,
                                       trainable=is_training)        
        feat = tf.multiply(weight_scale,
                           tf.math.l2_normalize(feat, axis=-1, epsilon=1e-12))

      classes.append(class_graph_fn(feat, num_classes, num, name))
      bboxes.append(bbox_graph_fn(feat, num, name))

    classes = tf.concat(classes, axis=1)
    bboxes = tf.concat(bboxes, axis=1)

    return classes, bboxes

def hard_negative_mining(logits_classes, gt_mask):
  # compute mask and index for foregound objects
  fg_mask = tf.to_float(tf.math.equal(gt_mask, 1))
  fg_index = tf.where(tf.math.equal(gt_mask, 1))

  # decide number of samples
  fg_num = tf.to_int32(tf.reduce_sum(fg_mask))
  bg_num = tf.math.minimum(tf.shape(fg_mask)[0] - fg_num, fg_num * 3)

  # compute index for background object (class = 0)
  bg_score = tf.nn.softmax(logits_classes)[:, 0]
  bg_score = tf.multiply(bg_score, 1 - fg_mask) + fg_mask
  bg_score = tf.multiply(-1.0, bg_score)
  bg_v, bg_index = tf.math.top_k(bg_score, k=bg_num)

  return fg_index, bg_index

def heuristic_sampling(gt_mask):
  # Balance & Sub-sample fg and bg objects
  fg_ids = np.where(gt_mask == 1)[0]
  fg_extra = (len(fg_ids) -
              int(math.floor(TRAIN_SAMPLES_PER_IMAGE * TRAIN_FG_RATIO)))
  if fg_extra > 0:
    random_fg_ids = np.random.choice(fg_ids, fg_extra, replace=False)
    gt_mask[random_fg_ids] = 0

  bg_ids = np.where(gt_mask == -1)[0]
  bg_extra = len(bg_ids) - (TRAIN_SAMPLES_PER_IMAGE - np.sum(gt_mask == 1))
  if bg_extra > 0:
    random_bg_ids = np.random.choice(bg_ids, bg_extra, replace=False)
    gt_mask[random_bg_ids] = 0

  fg_index = np.where(gt_mask == 1)[0]
  bg_index = np.where(gt_mask == -1)[0]
  return fg_index, bg_index

def loss(inputs, outputs, class_weights, bboxes_weights):
  gt_classes = inputs[2]
  gt_bboxes = inputs[3]
  gt_mask = inputs[4]
  feat_classes = outputs[0]
  feat_bboxes = outputs[1]

  gt_mask = tf.reshape(gt_mask, [-1])
  logits_classes = tf.reshape(feat_classes, [-1, tf.shape(feat_classes)[2]])
  gt_classes = tf.reshape(gt_classes, [-1, 1])
  logits_bboxes = tf.reshape(feat_bboxes, [-1, 4])
  gt_bboxes = tf.reshape(gt_bboxes, [-1, 4])

  # # heuristic sampling
  # fg_index, bg_index = tf.py_func(
  #   heuristic_sampling, [gt_mask], (tf.int64, tf.int64))

  # hard negative mining
  fg_index, bg_index = hard_negative_mining(logits_classes, gt_mask)

  loss_classes = class_weights * create_loss_classes_fn(logits_classes, gt_classes, fg_index, bg_index)

  loss_bboxes = bboxes_weights * create_loss_bboxes_fn(logits_bboxes, gt_bboxes, fg_index)

  return loss_classes, loss_bboxes
