"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""

import tensorflow as tf

from network.external.tf_slim import resnet_v2

slim = tf.contrib.slim


def net(inputs, num_classes, is_training, data_format="channels_first"):
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits, end_points = resnet_v2.resnet_v2_32(inputs,
                                                num_classes,
                                                is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["predictions"]
    }

    return logits, predictions
