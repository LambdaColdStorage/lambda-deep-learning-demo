"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""

import tensorflow as tf

from source.network.external.tf_slim import inception_v4

slim = tf.contrib.slim


def net(inputs, num_classes, is_training, data_format="channels_first"):
  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    logits, end_points = inception_v4.inception_v4(inputs,
                                                   num_classes,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions
