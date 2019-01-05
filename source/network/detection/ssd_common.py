import math
import numpy as np

import tensorflow as tf

from source.network.detection import detection_common

KERNEL_INIT = tf.contrib.layers.xavier_initializer()

# num_anchors = []
# anchors_map = None
anchors_stride = [8, 16, 32, 64, 128, 256, 512]
anchors_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
# control the size of the default square priorboxes
# REF: https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp#L164
min_ratio = 10
max_ratio = 90
min_dim = 512
TRAIN_FG_IOU = 0.5
TRAIN_BG_IOU = 0.5

priorvariance = [0.1, 0.1, 0.2, 0.2]

RESULT_SCORE_THRESH = 0.01
RESULTS_PER_IM = 200
NMS_THRESH = 0.45
NUM_CLASSES = 81

def ssd_priorbox_layer(min_dim, aspect_ratio, step, min_size, max_size):
  feat_map_size = min_dim / step
  num_boxes = len(aspect_ratio) * 2 + 2

  # create tensors for x_min, y_min, x_max, y_max
  x_min = np.zeros((feat_map_size, feat_map_size, num_boxes), dtype=np.float32)
  y_min = np.zeros_like(x_min, dtype=np.float32)
  x_max = np.zeros_like(x_min, dtype=np.float32)
  y_max = np.zeros_like(x_max, dtype=np.float32)

  # The bigger priorbox has size of sqrt(min_size * max_size)
  list_w = [min_size, math.sqrt(min_size * max_size)]
  list_h =[min_size, math.sqrt(min_size * max_size)]

  for r in aspect_ratio:
    l_s = min_size * math.sqrt(r)
    s_s = min_size / math.sqrt(r)
    list_w.append(l_s)
    list_h.append(s_s)
    list_w.append(s_s)
    list_h.append(l_s)

  num_anchors = len(list_w)

  list_x = np.arange(0, min_dim, step) + step / 2
  list_y = list_x
  x_cen_v, y_cen_v = np.meshgrid(list_x, list_y)


  for i in range(len(list_w)):
    x_min[:, :, i] = x_cen_v - list_w[i] / 2
    x_max[:, :, i] = x_cen_v + list_w[i] / 2
    y_min[:, :, i] = y_cen_v - list_h[i] / 2
    y_max[:, :, i] = y_cen_v + list_h[i] / 2


  x_min = x_min.reshape((-1, 1))
  x_max = x_max.reshape((-1, 1))
  y_min = y_min.reshape((-1, 1))
  y_max = y_max.reshape((-1, 1))

  priorbox = np.concatenate((x_min, y_min, x_max, y_max), axis=1)
  priorbox = priorbox / min_dim

  return priorbox, num_anchors


def ssd_create_priorbox(min_dim, aspect_ratios, steps, min_sizes, max_sizes):
  list_priorbox = []
  list_num_anchors = []
  for i_layer in range(len(aspect_ratios)):
    aspect_ratio = aspect_ratios[i_layer]
    step = steps[i_layer]
    min_size = min_sizes[i_layer]
    max_size = max_sizes[i_layer]
    priorbox, num_anchors = ssd_priorbox_layer(
      min_dim,
      aspect_ratio,
      step,
      min_size,
      max_size)
    list_priorbox.append(priorbox)
    list_num_anchors.append(num_anchors)
  return list_priorbox, list_num_anchors


def get_anchors():
  step = int(math.floor((max_ratio - min_ratio) / (len(anchors_aspect_ratios) - 2)))
  min_sizes = []
  max_sizes = []
  for ratio in xrange(min_ratio, max_ratio + 1, step):
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * (ratio + step) / 100.)
  min_sizes = [min_dim * 4 / 100.] + min_sizes
  max_sizes = [min_dim * 10 / 100.] + max_sizes
  min_sizes = [math.floor(x) for x in min_sizes]
  max_sizes = [math.floor(x) for x in max_sizes]

  list_priorbox, list_num_anchors = ssd_create_priorbox(
    min_dim,
    anchors_aspect_ratios,
    anchors_stride,
    min_sizes,
    max_sizes)
  anchors_map = np.concatenate(list_priorbox, axis=0)
  num_anchors = list_num_anchors

  return anchors_map, num_anchors


def encode_gt(labels, boxes, anchors_map, batch_size):

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
    ret_iou = detection_common.np_iou(anchors_map, b)

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

    fg_idx = np.where(max_iou > TRAIN_FG_IOU)[0]
    bg_idx = np.where(max_iou < TRAIN_BG_IOU)[0]
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
      gt_bboxes, anchors_map)

    # scale with variance 
    cx, cy, w, h = tf.unstack(gt_bboxes, 4, axis=1)
    cx = tf.scalar_mul(1.0 / priorvariance[0], cx)
    cy = tf.scalar_mul(1.0 / priorvariance[1], cy)
    w = tf.scalar_mul(1.0 / priorvariance[2], w)
    h = tf.scalar_mul(1.0 / priorvariance[3], h)
    gt_bboxes = tf.concat([tf.expand_dims(cx, -1),
                           tf.expand_dims(cy, -1),
                           tf.expand_dims(w, -1),
                           tf.expand_dims(h, -1)], axis=1)

    return gt_labels, gt_bboxes, gt_masks 

  list_labels = tf.unstack(labels, num=batch_size)
  list_boxes = tf.unstack(boxes, num=batch_size)

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


def decode_bboxes(box_predictions, anchors):
  """
  Args:
      box_predictions: (..., 4), logits
      anchors: (..., 4), floatbox. Must have the same shape
  Returns:
      box_decoded: (..., 4), float32. With the same shape.
  """
  priorvariance = [0.1, 0.2]
  orig_shape = tf.shape(anchors)
  box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
  box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)

  anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
  anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

  waha = anchors_x2y2 - anchors_x1y1
  xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

  wbhb = tf.exp(box_pred_twth * priorvariance[1]) * waha
  xbyb = box_pred_txty * priorvariance[0] * waha + xaya
  x1y1 = xbyb - wbhb * 0.5
  x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
  out = tf.concat([x1y1, x2y2], axis=-2)
  return tf.reshape(out, orig_shape)

def decode_bboxes_batch(boxes, anchors, batch_size):
  boxes = tf.unstack(boxes, batch_size)
  boxes = [decode_bboxes(boxes_per_img, anchors) for boxes_per_img in boxes]
  boxes = tf.stack(boxes)
  return boxes


def detect(scores, bboxes, anchors_map):
  # Non-Max Surpression applied to each class separately
  # Background class is not cosidered in detection

  def nms_bboxes(prob, box, idx, anchors):
    """
    Args:
        prob: n probabilities
        box: n x 4 boxes
    Returns:
        n boolean, the selection
    """
    # filter by score threshold

    ids = tf.reshape(tf.where(tf.reshape(prob, [-1]) > RESULT_SCORE_THRESH), [-1])
    prob = tf.reshape(tf.gather(prob, ids), [-1])
    box = tf.gather(box, ids)
    idx = tf.reshape(tf.gather(idx, ids), [-1])
    anchors = tf.gather(anchors, ids)


    # NMS within each class
    mask = tf.image.non_max_suppression(
        box, prob, RESULTS_PER_IM, NMS_THRESH)

    prob = tf.gather(prob, mask)
    box = tf.gather(box, mask)
    idx = tf.gather(idx, mask)
    anchors = tf.gather(anchors, mask)
    return prob, idx, box, anchors

  list_scores = tf.split(axis=1, num_or_size_splits=NUM_CLASSES, value=scores)

  topk_scores = None
  topk_labels = None
  topk_labels = None
  topk_anchors = None

  for i in range(NUM_CLASSES):
    if i > 0:
      class_id = tf.scalar_mul(i, tf.ones_like(list_scores[i], dtype=tf.int32))
      if i == 1:
        topk_scores, topk_labels, topk_bboxes, topk_anchors = nms_bboxes(list_scores[i], bboxes, class_id, anchors_map)
      else:
        s, l, b, a = nms_bboxes(list_scores[i], bboxes, class_id, anchors_map)
        topk_scores = tf.concat([topk_scores, s], axis=0)
        topk_labels = tf.concat([topk_labels, l], axis=0)
        topk_bboxes = tf.concat([topk_bboxes, b], axis=0)
        topk_anchors = tf.concat([topk_anchors, a], axis=0)

  # sort the result
  topk_scores, ids = tf.nn.top_k(topk_scores, k=tf.math.minimum(tf.size(topk_scores), RESULTS_PER_IM))
  topk_bboxes = tf.gather(topk_bboxes, ids)
  topk_labels = tf.gather(topk_labels, ids)
  topk_anchors = tf.gather(topk_anchors, ids)
  return topk_scores, topk_labels, topk_bboxes, topk_anchors


def detect_batch(scores, bboxes, anchors_map, batch_size):
  scores = tf.unstack(scores, batch_size)
  bboxes = tf.unstack(bboxes, batch_size)

  detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors = [], [], [], []

  for scores_per_image, bboxes_per_image in zip(scores, bboxes):
    detection_topk_scores_per_image, detection_topk_labels_per_image, detection_topk_bboxes_per_image, detection_topk_anchors_per_image = detect(
      scores_per_image, bboxes_per_image, anchors_map)
    detection_topk_scores.append(detection_topk_scores_per_image)
    detection_topk_labels.append(detection_topk_labels_per_image)
    detection_topk_bboxes.append(detection_topk_bboxes_per_image)
    detection_topk_anchors.append(detection_topk_anchors_per_image)

  return detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors
