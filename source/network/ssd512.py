import tensorflow as tf


def ssd_feature_fn(last_layer, feats, backbone_output_layer):
  # # Shared SSD feature layer

  # print(feats)
  # print(feats)
  # output_backbone = feats[backbone_output_layer]
  output_backbone = last_layer

  # Add additional feature layers
  kernel_init = tf.contrib.layers.xavier_initializer()

  net = tf.layers.conv2d(inputs=output_backbone,
                         filters=1024,
                         kernel_size=[3, 3],
                         strides=(1, 1),
                         padding=('SAME'),
                         dilation_rate=6,
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv6')

  net = tf.layers.conv2d(inputs=net,
                         filters=1024,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv7')
  feats["ssd_conv7"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv8_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=512,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv8_2')
  feats["ssd_conv8_2"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=128,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv9_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv9_2')
  feats["ssd_conv9_2"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=128,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv10_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv10_2')
  feats["ssd_conv10_2"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=128,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv11_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv11_2')
  feats["ssd_conv11_2"] = net

  net = tf.layers.conv2d(inputs=net,
                         filters=128,
                         kernel_size=[1, 1],
                         strides=(1, 1),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv12_1')

  net = tf.layers.conv2d(inputs=net,
                         filters=256,
                         kernel_size=[3, 3],
                         strides=(2, 2),
                         padding=('SAME'),
                         kernel_initializer=kernel_init,
                         activation=tf.nn.relu,
                         name='conv12_2')
  feats["ssd_conv12_2"] = net

  return feats


def class_graph_fn(feat, num_classes, num_anchors):
  data_format = 'channels_last'
  kernel_init = tf.variance_scaling_initializer()
  output = tf.layers.conv2d(inputs=feat,
                            filters=num_anchors * num_classes,
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
  output = tf.layers.conv2d(inputs=feat,
                            filters=num_anchors * 4,
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
  loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
    logits=logits,
    labels=labels))
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


def net(last_layer, feats,
        backbone_output_layer,
        feature_layers,
        num_classes, num_anchors,
        is_training, data_format="channels_last"):

  
  with tf.variable_scope(name_or_scope='SSD',
                         values=[feats],
                         reuse=tf.AUTO_REUSE):

    # Add shared features
    feats = ssd_feature_fn(last_layer, feats, backbone_output_layer)

    classes = []
    bboxes = []
    for layer, num in zip(feature_layers, num_anchors):
      feat = feats[layer]                 
      
      # # According to author's paper, only do it on conv4_3 with learnable scale
      # if layer == "vgg_16/conv4/conv4_3":
      #   weight_scale = tf.Variable([20.] * 512, trainable=is_training, name='l2_norm_scaler')
      #   feat = tf.multiply(weight_scale,
      #                      tf.math.l2_normalize(feat, axis=-1, epsilon=1e-12))

      classes.append(class_graph_fn(feat, num_classes, num))

      # Do it for all bboxes layers, with no learnable scale
      feat = tf.math.l2_normalize(feat,
                                  axis=-1,
                                  epsilon=1e-12)
      bboxes.append(bbox_graph_fn(feat, num))

    classes = tf.concat(classes, axis=1)
    bboxes = tf.concat(bboxes, axis=1)

    return classes, bboxes


def loss(inputs, outputs, class_weights, bboxes_weights):
  gt_classes = inputs[2]
  gt_bboxes = inputs[3]
  gt_mask = inputs[4]
  feat_classes = outputs[0]
  feat_bboxes = outputs[1]

  loss_classes = class_weights * create_loss_classes_fn(feat_classes, gt_classes, gt_mask)

  loss_bboxes = bboxes_weights * create_loss_bboxes_fn(feat_bboxes, gt_bboxes, gt_mask)

  return loss_classes, loss_bboxes