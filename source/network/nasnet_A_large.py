"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""

import tensorflow as tf

from source.network.external.tf_slim import nasnet

slim = tf.contrib.slim


def net(inputs, num_classes, is_training, data_format="channels_first"):
  with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
    logits, end_points = nasnet.build_nasnet_large(inputs,
                                                   num_classes,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions
