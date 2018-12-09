import numpy as np
import math

import tensorflow as tf

TRAIN_SAMPLES_PER_IMAGE = 512
TRAIN_FG_RATIO = 0.5
KERNEL_INIT = tf.contrib.layers.xavier_initializer()


def vgg_mod_fn(last_layer, feats, pre_weights):
  # Shared SSD feature layer

  output_backbone = last_layer

  # Add modified VGG layers 
  init_ssd_conv6_w = tf.constant_initializer(pre_weights["mod_w_fc6"])
  init_ssd_conv6_b = tf.constant_initializer(pre_weights["mod_b_fc6"])

  net = tf.layers.conv2d(inputs=output_backbone,
                         filters=1024,
                         kernel_size=[3, 3],
                         strides=(1, 1),
                         padding=('SAME'),
                         dilation_rate=6,
                         kernel_initializer=init_ssd_conv6_w,
                         bias_initializer=init_ssd_conv6_b,
                         activation=tf.nn.relu,
                         name='conv6')
  feats["ssd_conv6"] = net

  init_ssd_conv7_w = tf.constant_initializer(pre_weights["mod_w_fc7"])
  init_ssd_conv7_b = tf.constant_initializer(pre_weights["mod_b_fc7"])

  net = tf.layers.conv2d(inputs=net,
                         filters=1024,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=init_ssd_conv7_w,
                         bias_initializer=init_ssd_conv7_b,
                         activation=tf.nn.relu,
                         name='conv7')
  feats["ssd_conv7"] = net

  return net, feats

def ssd_feature_fn(last_layer, feats):


  # Add additional convolutional layers

  net = tf.layers.conv2d(inputs=last_layer,
                         filters=256,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv8_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=512,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv8_2')
  feats["ssd_conv8_2"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=128,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv9_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv9_2')
  feats["ssd_conv9_2"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=128,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv10_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv10_2')
  feats["ssd_conv10_2"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=128,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv11_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv11_2')
  feats["ssd_conv11_2"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=128,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv12_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=KERNEL_INIT,
                         activation=tf.nn.relu,
                         name='conv12_2')
  feats["ssd_conv12_2"] = net

  return feats


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


def net(last_layer, feats, pre_weights,
        feature_layers,
        num_classes, num_anchors,
        is_training, data_format="channels_last"):

  with tf.variable_scope(name_or_scope='MOD',
                         values=[last_layer, feats],
                         reuse=tf.AUTO_REUSE):
    # Add shared features
    net, feats = vgg_mod_fn(last_layer, feats, pre_weights)

  with tf.variable_scope(name_or_scope='SSD',
                         values=[net, feats],
                         reuse=tf.AUTO_REUSE):

    # Add shared features
    feats = ssd_feature_fn(net, feats)

    classes = []
    bboxes = []
    for layer, num in zip(feature_layers, num_anchors):
      feat = feats[layer]                 
      
      # According to the original SSD paper, normalize conv4_3 with learnable scale
      # In pratice doing so indeed reduce the classification loss significantly
      if layer == "vgg_16/conv4/conv4_3":
        weight_scale = tf.Variable([20.] * 512, trainable=is_training, name='l2_norm_scaler')
        feat = tf.multiply(weight_scale,
                           tf.math.l2_normalize(feat, axis=-1, epsilon=1e-12))

      classes.append(class_graph_fn(feat, num_classes, num, layer))

      # Different from the original SSD paper, we normalize the feature maps for all bbox layers
      # Otherwise we found bbox loss will be too high
      feat = tf.math.l2_normalize(feat, axis=-1, epsilon=1e-12)

      bboxes.append(bbox_graph_fn(feat, num, layer))

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