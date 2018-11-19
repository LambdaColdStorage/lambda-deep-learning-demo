import tensorflow as tf

# TODO: make a class

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

def class_graph_fn(feat, num_classes):
  data_format = 'channels_last'
  kernel_init = tf.variance_scaling_initializer()
  output = tf.layers.conv2d(inputs=feat,
                            filters= 5 * 3 * num_classes,
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


def bbox_graph_fn(feat):
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
                      [tf.shape(output)[0],
                       -1,
                       4],
                      name='feat_bboxes')    
  return output

def create_loss_classes_fn(feat_classes, gt_classes, mask):
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

def create_loss_bboxes_fn(feat_bboxes, gt_bboxes, mask):
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

def net(inputs, num_classes,
        is_training, data_format="channels_last"):

  # Shared SSD feature layer
  feat_vgg = inputs[0]

  with tf.variable_scope(name_or_scope='SSD',
                         values=[inputs],
                         reuse=tf.AUTO_REUSE):

    feat_ssd = ssd_feature_fn(feat_vgg)

    # Class head
    feat_classes = class_graph_fn(feat_ssd, num_classes)

    # BBox head
    feat_bboxes = bbox_graph_fn(feat_ssd)

    return feat_classes, feat_bboxes

def loss(inputs, outputs):
  gt_classes = inputs[1]
  gt_bboxes = inputs[2]
  gt_mask = inputs[3]
  feat_classes = outputs[0]
  feat_bboxes = outputs[1]

  mask = tf.math.not_equal(gt_mask, 0)
  mask.set_shape([None])

  loss_classes = create_loss_classes_fn(feat_classes, gt_classes, mask)

  loss_bboxes = create_loss_bboxes_fn(feat_bboxes, gt_bboxes, mask)

  loss = loss_classes + loss_bboxes 

  return loss