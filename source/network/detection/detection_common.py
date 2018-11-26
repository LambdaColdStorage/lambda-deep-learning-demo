import numpy as np

from pycocotools.mask import iou

import tensorflow as tf


def np_iou(A, B):
  def to_xywh(box):
    box = box.copy()
    box[:, 2] -= box[:, 0]
    box[:, 3] -= box[:, 1]
    return box

  ret = iou(
    to_xywh(A), to_xywh(B),
    np.zeros((len(B),), dtype=np.bool))
  return ret


def encode_bbox_target(boxes, anchors):
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
  # each is (...)x1x2
  anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
  anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

  waha = anchors_x2y2 - anchors_x1y1
  xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

  # TODO: change 512 to config.resolution
  clip = np.log(512 / 16.)
  wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
  xbyb = box_pred_txty * waha + xaya
  x1y1 = xbyb - wbhb * 0.5
  x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
  out = tf.concat([x1y1, x2y2], axis=-2)
  return tf.reshape(out, orig_shape)


def decode_bboxes_batch(boxes, anchors):
  # TODO: implement decode function
  boxes = tf.unstack(boxes)
  boxes = [decode_bboxes(boxes_per_img, anchors) for boxes_per_img in boxes]
  boxes = tf.stack(boxes)
  return boxes


def detect(scores, bboxes, config, anchors_map):

  def nms_bboxes(prob, box, idx, anchors):
    """
    Args:
        prob: n probabilities
        box: n x 4 boxes
    Returns:
        n boolean, the selection
    """
    output_shape = tf.shape(prob)

    # filter by score threshold
    ids = tf.reshape(tf.where(prob > config.RESULT_SCORE_THRESH), [-1])
    prob = tf.gather(prob, ids)
    box = tf.gather(box, ids)
    idx = tf.gather(idx, ids)
    anchors = tf.gather(anchors, ids)

    # NMS within each class
    mask = tf.image.non_max_suppression(
        box, prob, config.RESULTS_PER_IM, config.NMS_THRESH)

    # ids = tf.to_int32(tf.gather(ids, mask))
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


def detect_batch(scores, bboxes, config, anchors_map):
  scores = tf.unstack(scores)
  bboxes = tf.unstack(bboxes)

  detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors = [], [], [], []

  for scores_per_image, bboxes_per_image in zip(scores, bboxes):
    detection_topk_scores_per_image, detection_topk_labels_per_image, detection_topk_bboxes_per_image, detection_topk_anchors_per_image = detect(
      scores_per_image, bboxes_per_image, config, anchors_map)
    detection_topk_scores.append(detection_topk_scores_per_image)
    detection_topk_labels.append(detection_topk_labels_per_image)
    detection_topk_bboxes.append(detection_topk_bboxes_per_image)
    detection_topk_anchors.append(detection_topk_anchors_per_image)

  return detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors

