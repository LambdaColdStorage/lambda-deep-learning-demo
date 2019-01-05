import math
import numpy as np

import tensorflow as tf

from source.network.detection import detection_common

KERNEL_INIT = tf.contrib.layers.xavier_initializer()

PRIOR_VARIANCE = [0.1, 0.1, 0.2, 0.2]

# Non Maximum Supression
RESULT_SCORE_THRESH = 0.01
RESULTS_PER_IM = 200
NMS_THRESH = 0.45

# HARD NEGATIVE MINING
HARD_MINING_FG_IOU = 0.5
HARD_MINING_BG_IOU = 0.5

# Heuristic sampling
HEURISTIC_MINING_SAMPLES_PER_IMAGE = 512
HEURISTIC_MINING_FG_RATIO = 0.5

# ------------------------------------------------------------------------
# Priorbox
# ------------------------------------------------------------------------
def ssd_priorbox_layer(min_dim, aspect_ratio, step, min_size, max_size):
  # Match official SSD feature map size
  feat_map_size = int(math.ceil(float(min_dim) / step))
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


def get_anchors(anchors_stride,
                anchors_aspect_ratios,
                min_size_ratio,
                max_size_ratio,
                input_dim):
  step = int(math.floor((max_size_ratio - min_size_ratio) / (len(anchors_aspect_ratios) - 2)))
  min_sizes = []
  max_sizes = []
  for ratio in xrange(min_size_ratio, max_size_ratio + 1, step):
    min_sizes.append(input_dim * ratio / 100.)
    max_sizes.append(input_dim * (ratio + step) / 100.)
  min_sizes = [input_dim * 4 / 100.] + min_sizes
  max_sizes = [input_dim * 10 / 100.] + max_sizes
  min_sizes = [math.floor(x) for x in min_sizes]
  max_sizes = [math.floor(x) for x in max_sizes]

  list_priorbox, list_num_anchors = ssd_create_priorbox(
    input_dim,
    anchors_aspect_ratios,
    anchors_stride,
    min_sizes,
    max_sizes)
  anchors_map = np.concatenate(list_priorbox, axis=0)
  num_anchors = list_num_anchors

  return anchors_map, num_anchors


# ------------------------------------------------------------------------
# Encode/Decode
# ------------------------------------------------------------------------
def encode_bbox(boxes, anchors):
  """
  Args:
      boxes: (..., 4), float32
      anchors: (..., 4), float32
  Returns:
      box_encoded: (..., 4), float32 with the same shape.
  """
  anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
  anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
  waha = anchors_x2y2 - anchors_x1y1
  xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

  boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
  boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
  wbhb = boxes_x2y2 - boxes_x1y1
  xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

  # Note that here not all boxes are valid. Some may be zero
  txty = (xbyb - xaya) / waha
  twth = tf.log(wbhb / waha)  # may contain -inf for invalid boxes
  encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)
  return tf.reshape(encoded, tf.shape(boxes))


def decode_bboxes(box_predictions, anchors):
  """
  Args:
      box_predictions: (..., 4), logits
      anchors: (..., 4), floatbox. Must have the same shape
  Returns:
      box_decoded: (..., 4), float32. With the same shape.
  """
  orig_shape = tf.shape(anchors)
  box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
  box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)

  anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
  anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

  waha = anchors_x2y2 - anchors_x1y1
  xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

  wbhb = tf.exp(box_pred_twth * PRIOR_VARIANCE[2]) * waha
  xbyb = box_pred_txty * PRIOR_VARIANCE[0] * waha + xaya
  x1y1 = xbyb - wbhb * 0.5
  x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
  out = tf.concat([x1y1, x2y2], axis=-2)
  return tf.reshape(out, orig_shape)


def decode_bboxes_batch(boxes, anchors, batch_size):
  boxes = tf.unstack(boxes, batch_size)
  boxes = [decode_bboxes(boxes_per_img, anchors) for boxes_per_img in boxes]
  boxes = tf.stack(boxes)
  return boxes


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

    fg_idx = np.where(max_iou > HARD_MINING_FG_IOU)[0]
    bg_idx = np.where(max_iou < HARD_MINING_BG_IOU)[0]
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
    gt_bboxes = encode_bbox(gt_bboxes, anchors_map)

    # scale with variance 
    cx, cy, w, h = tf.unstack(gt_bboxes, 4, axis=1)
    cx = tf.scalar_mul(1.0 / PRIOR_VARIANCE[0], cx)
    cy = tf.scalar_mul(1.0 / PRIOR_VARIANCE[1], cy)
    w = tf.scalar_mul(1.0 / PRIOR_VARIANCE[2], w)
    h = tf.scalar_mul(1.0 / PRIOR_VARIANCE[3], h)
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


# ------------------------------------------------------------------------
# SSD building blocks
# ------------------------------------------------------------------------
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


# def ssd_block(outputs, name, data_format, conv_strides, filter_size, num_filters):

#     num_conv = len(conv_strides)
#     for i in range(num_conv):
#         stride = conv_strides[i]
#         w = filter_size[i]
#         num_filter = num_filters[i]

#         if stride == 2:
#             # Use customized padding when stride == 2
#             # https://stackoverflow.com/questions/42924324/tensorflows-asymmetric-padding-assumptions
#             if data_format == "channels_last":
#               outputs = tf.pad(outputs, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
#             else:
#               outputs = tf.pad(outputs, [[0, 0], [0, 0], [1, 0], [1, 0]], "CONSTANT")
#             padding_strategy = "VALID"
#         elif w == 4:
#             # Use customized padding when for the last ssd feature layer (filter_size == 4)
#             if data_format == "channels_last":
#               outputs = tf.pad(outputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
#             else:
#               outputs = tf.pad(outputs, [[0, 0], [0, 0], [1, 1], [1, 1]], "CONSTANT")
#             padding_strategy = "VALID"
#         else:
#             padding_strategy = "SAME"
#         outputs = tf.layers.conv2d(
#                 outputs,
#           filters=num_filter,
#           kernel_size=(w, w),
#           strides=(stride, stride),
#           padding=(padding_strategy),
#           data_format=data_format,
#           kernel_initializer=KERNEL_INIT,
#           activation=tf.nn.relu,
#           name=name + "_" + str(i + 1))
#     return outputs


def ssd_block(outputs, name, data_format, conv_strides, filter_size, num_filters, padding):
  # Result will be slightly different to caffe due to TF's asymmetric padding
  # https://stackoverflow.com/questions/42924324/tensorflows-asymmetric-padding-assumptions

    num_conv = len(conv_strides)
    for i in range(num_conv):
        stride = conv_strides[i]
        w = filter_size[i]
        num_filter = num_filters[i]
        outputs = tf.layers.conv2d(
                outputs,
          filters=num_filter,
          kernel_size=(w, w),
          strides=(stride, stride),
          padding=(padding[i]),
          data_format=data_format,
          kernel_initializer=KERNEL_INIT,
          activation=tf.nn.relu,
          name=name + "_" + str(i + 1))
    return outputs


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
              int(math.floor(HEURISTIC_MINING_SAMPLES_PER_IMAGE * HEURISTIC_MINING_FG_RATIO)))

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


def loss(gt, outputs, class_weights, bboxes_weights):
  gt_classes, gt_bboxes, gt_mask = gt
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


# ------------------------------------------------------------------------
# Detection head
# ------------------------------------------------------------------------
def detect_per_class(scores, bboxes, anchors_map, num_classes):
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

  list_scores = tf.split(axis=1, num_or_size_splits=num_classes, value=scores)

  topk_scores = None
  topk_labels = None
  topk_labels = None
  topk_anchors = None

  for i in range(num_classes):
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


def detect_joint_classes(scores, bboxes, anchors_map, num_classes):
  # Non-Max surpression is applied to all classes together
  # Also remove the background classes for better mAP

  def nms_bboxes(prob, box, idx, anchors):
    """
    Args:
        prob: n probabilities
        box: n x 4 boxes
    Returns:
        n boolean, the selection
    """

    # filter by score threshold
    ids = tf.reshape(tf.where(prob > RESULT_SCORE_THRESH), [-1])
    prob = tf.gather(prob, ids)
    box = tf.gather(box, ids)
    idx = tf.gather(idx, ids)
    anchors = tf.gather(anchors, ids)

    # NMS within each class
    mask = tf.image.non_max_suppression(
        box, prob, RESULTS_PER_IM, NMS_THRESH)

    prob = tf.gather(prob, mask)
    box = tf.gather(box, mask)
    idx = tf.gather(idx, mask)
    anchors = tf.gather(anchors, mask)

    # sort the result
    prob, ids = tf.nn.top_k(prob, k=tf.size(prob))
    box = tf.gather(box, ids)
    idx = tf.gather(idx, ids)
    anchors = tf.gather(anchors, ids)
    return prob, idx, box, anchors

  # HACK: make background class zero score
  scores_list = tf.split(axis=1, num_or_size_splits=num_classes, value=scores)
  scores_list[0] = scores_list[0] * 0
  scores = tf.concat(axis=1, values=scores_list)

  best_classes = tf.argmax(scores, axis=1)
  idx = tf.stack(
    [tf.dtypes.cast(tf.range(tf.shape(scores)[0]), tf.int64), best_classes],
    axis=1)
  best_scores = tf.gather_nd(scores, idx)

  # only collect the foreground objects
  ids = tf.reshape(tf.where(best_classes > 0), [-1])
  best_scores = tf.gather(best_scores, ids)
  bboxes = tf.gather(bboxes, ids)
  anchors = tf.gather(anchors_map, ids)
  best_classes = tf.gather(best_classes, ids)

  topk_scores, topk_labels, topk_bboxes, topk_anchors = nms_bboxes(best_scores, bboxes, best_classes, anchors)

  return topk_scores, topk_labels, topk_bboxes, topk_anchors


def detect_batch(scores, bboxes, anchors_map, batch_size, num_classes):
  scores = tf.unstack(scores, batch_size)
  bboxes = tf.unstack(bboxes, batch_size)

  detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors = [], [], [], []

  for scores_per_image, bboxes_per_image in zip(scores, bboxes):
    detection_topk_scores_per_image, detection_topk_labels_per_image, detection_topk_bboxes_per_image, detection_topk_anchors_per_image = detect_per_class(
      scores_per_image, bboxes_per_image, anchors_map, num_classes)
    detection_topk_scores.append(detection_topk_scores_per_image)
    detection_topk_labels.append(detection_topk_labels_per_image)
    detection_topk_bboxes.append(detection_topk_bboxes_per_image)
    detection_topk_anchors.append(detection_topk_anchors_per_image)

  return detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors