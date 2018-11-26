import tensorflow as tf


def ssd_feature_fn(feat):
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

def class_graph_fn(feat, num_classes, num_anchors):
  data_format = 'channels_last'
  kernel_init = tf.variance_scaling_initializer()
  output = tf.layers.conv2d(inputs=feat,
                            filters= num_anchors * num_classes,
                            kernel_size=[3, 3],
                            strides=(1, 1),
                            padding=('SAME'),
                            data_format=data_format,
                            kernel_initializer=kernel_init,
                            activation=None)
  output = tf.reshape(output,
                      [tf.shape(output)[0],
                       -1,
                       num_classes],
                      name='feat_classes')
  return output


def bbox_graph_fn(feat, num_anchors):
  data_format = 'channels_last'
  kernel_init = tf.variance_scaling_initializer()
  output = tf.math.l2_normalize(feat,
                                axis=-1,
                                epsilon=1e-12)
  output = tf.layers.conv2d(inputs=output,
                            filters= num_anchors * 4,
                            kernel_size=[3, 3],
                            strides=(1, 1),
                            padding=('SAME'),
                            data_format=data_format,
                            kernel_initializer=kernel_init,
                            activation=None)
  output = tf.reshape(output,
                      [tf.shape(output)[0],
                       -1,
                       4],
                      name='feat_bboxes')
  return output

def create_loss_classes_fn(feat_classes, gt_classes, gt_mask):
  mask = tf.math.not_equal(gt_mask, 0)
  mask.set_shape([None])  
  logits = tf.boolean_mask(
    tf.reshape(feat_classes, [-1, tf.shape(feat_classes)[2]]),
    mask)
  labels = tf.boolean_mask(
    tf.reshape(gt_classes, [-1, 1]),
    mask)
  loss = tf.losses.sparse_softmax_cross_entropy(
    logits=logits,
    labels=labels)
  return loss

def create_loss_bboxes_fn(feat_bboxes, gt_bboxes, gt_mask):
  mask = tf.math.equal(gt_mask, 1)
  mask.set_shape([None])   
  pred = tf.boolean_mask(
    tf.reshape(feat_bboxes, [-1, 4]),
    mask)
  gt = tf.boolean_mask(
    tf.reshape(gt_bboxes, [-1, 4]),
    mask)
  abs_diff = tf.abs(pred - gt)
  minx = tf.minimum(abs_diff, 1)
  loss = tf.reduce_mean(0.5 * ((abs_diff - 1) * minx + abs_diff))
  return loss

def net(inputs, num_classes, num_anchors,
        is_training, data_format="channels_last"):

  # Shared SSD feature layer
  feat_vgg = inputs[0]

  with tf.variable_scope(name_or_scope='SSD',
                         values=[inputs],
                         reuse=tf.AUTO_REUSE):

    feat_ssd = ssd_feature_fn(feat_vgg)

    # Class head
    feat_classes = class_graph_fn(feat_ssd, num_classes, num_anchors)

    # BBox head
    feat_bboxes = bbox_graph_fn(feat_ssd, num_anchors)

    return feat_classes, feat_bboxes

def loss(inputs, outputs, class_weights, bboxes_weights):
  gt_classes = inputs[1]
  gt_bboxes = inputs[2]
  gt_mask = inputs[3]
  feat_classes = outputs[0]
  feat_bboxes = outputs[1]

  loss_classes = create_loss_classes_fn(feat_classes, gt_classes, gt_mask)

  loss_bboxes = create_loss_bboxes_fn(feat_bboxes, gt_bboxes, gt_mask)

  loss = class_weights * loss_classes + bboxes_weights * loss_bboxes 

  return loss