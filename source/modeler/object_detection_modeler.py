"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import importlib
import numpy as np

import tensorflow as tf

from modeler import Modeler

# TODO: Put decode_bboxes and encode_bbox_target in a seperate file
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

def detect(scores, bboxes, config):

  def nms_bboxes(prob, box):
    """
    Args:
        prob: n probabilities
        box: nx4 boxes
    Returns:
        n boolean, the selection
    """
    output_shape = tf.shape(prob)

    # filter by score threshold
    ids = tf.reshape(tf.where(prob > config.RESULT_SCORE_THRESH), [-1])
    prob = tf.gather(prob, ids)
    box = tf.gather(box, ids)

    # NMS within each class
    mask = tf.image.non_max_suppression(
        box, prob, config.RESULTS_PER_IM, config.NMS_THRESH)

    ids = tf.to_int32(tf.gather(ids, mask))
    prob = tf.gather(prob, ids)
    box = tf.gather(box, ids)

    # sort the result
    prob, ids = tf.nn.top_k(prob, k=tf.size(prob))
    box = tf.gather(box, ids)
    return prob, box

  best_classes = tf.argmax(scores, axis=1)
  idx = tf.stack(
    [tf.dtypes.cast(tf.range(tf.shape(scores)[0]), tf.int64), best_classes],
    axis=1)
  best_scores = tf.gather_nd(scores, idx)

  topk_scores, topk_bboxes = nms_bboxes(best_scores, bboxes)

  return topk_scores, topk_bboxes

def detect_batch(scores, bboxes, config):
  scores = tf.unstack(scores)
  bboxes = tf.unstack(bboxes)

  detection_topk_scores, detection_topk_bboxes = [], []

  for scores_per_image, bboxes_per_image in zip(scores, bboxes):
    detection_topk_scores_per_image, detection_topk_bboxes_per_image = detect(
      scores_per_image, bboxes_per_image, config)
    detection_topk_scores.append(detection_topk_scores_per_image)
    detection_topk_bboxes.append(detection_topk_bboxes_per_image)

  return detection_topk_scores, detection_topk_bboxes


class ObjectDetectionModeler(Modeler):
  def __init__(self, args, net, loss):
    super(ObjectDetectionModeler, self).__init__(args, net)
    self.loss = loss
    self.feature_net = getattr(
      importlib.import_module("source.network." + self.config.feature_net),
      "net")
    self.feature_net_init_flag = True

    self.config.RESULT_SCORE_THRESH = 0.05
    self.config.RESULTS_PER_IM = 10
    self.config.NMS_THRESH = 0.5

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.anchors = inputter.get_anchors()
    self.anchors_map = inputter.get_anchors_map()

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
    self.gether_train_vars()
    
    return self.loss(inputs, outputs) 

  def create_detect_fn(self, outputs):
    # Args:
    #     outputs: 
    # Returns:
    #     detection_topk_scores: batch_size x (num_detections,), list of arrays
    #     detection_topk_indices: batch_size x (num_detections, 4), list of arrays
    feat_classes = outputs[0]
    feat_bboxes = outputs[1]

    score_classes = tf.nn.softmax(feat_classes)
    feat_bboxes = decode_bboxes_batch(feat_bboxes, self.anchors_map)

    detection_topk_scores, detection_topk_bboxes = detect_batch(score_classes, feat_bboxes, self.config)

    return detection_topk_scores, detection_topk_bboxes

  def model_fn(self, inputs):
    outputs = self.create_graph_fn(inputs[0])
    if self.config.mode == "train":
      loss = self.create_loss_fn(inputs, outputs)
      grads = self.create_grad_fn(loss)
      return {"loss": loss,
              "grads": grads,
              "learning_rate": self.learning_rate}
    elif self.config.mode == "infer":
      detection_scores, detection_bboxes = self.create_detect_fn(outputs)
      return {"scores": detection_scores,
              "bboxes": detection_bboxes,
              "images": inputs[0]} 

    # detection_masks, detection_bboxes = self.create_detect_fn(outputs)
    # return detection_masks, detection_bboxes

def build(args, network, loss):
  return ObjectDetectionModeler(args, network, loss)
