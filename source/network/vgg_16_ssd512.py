# Copyright 2018 Lambda Labs. All Rights Reserved.
# Licensed under
# ============================================================================

"""VGG 19 backbone for SSD based on Tensorflow slim
"""
from __future__ import print_function
import numpy as np

import tensorflow as tf

from source.network.external.tf_slim import vgg

slim = tf.contrib.slim


def net(inputs, pre_weights, data_format, is_training, init_flag, ckpt_path):

  (net, end_points) = vgg.vgg_16_ssd512(inputs, is_training=is_training)
  
  if init_flag:
    tf.logging.set_verbosity(tf.logging.WARN)
    restore_var_list = ["vgg_16"]
    variables_to_restore = {v.name.split(':')[0]: v
                            for v in tf.get_collection(
                                tf.GraphKeys.TRAINABLE_VARIABLES)}
    if restore_var_list is not None:
      variables_to_restore = {v: v for v in variables_to_restore
                              if any(x in v for x in restore_var_list)}

    print("Restoring weights from " + ckpt_path)
    tf.train.init_from_checkpoint(ckpt_path,
                                  variables_to_restore)
    init_flag = False
    print("Weights restored.")
    tf.logging.set_verbosity(tf.logging.INFO)

    w_fc6 = tf.get_default_graph().get_tensor_by_name('vgg_16/fc6/weights:0')
    b_fc6 = tf.get_default_graph().get_tensor_by_name('vgg_16/fc6/biases:0')
    w_fc7 = tf.get_default_graph().get_tensor_by_name('vgg_16/fc7/weights:0')
    b_fc7 = tf.get_default_graph().get_tensor_by_name('vgg_16/fc7/biases:0')


    # def create_session_config():
    #   """create session_config
    #   """
    #   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,
    #                               allow_growth=True)

    #   session_config = tf.ConfigProto(
    #     allow_soft_placement=True,
    #     log_device_placement=False,
    #     device_count={"GPU": 1},
    #     gpu_options=gpu_options)

    #   return session_config

    def create_session_config():
      """create session_config
      """
      session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_count={"CPU": 1})
      return session_config

    config = create_session_config()
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      _w_fc6, _b_fc6, _w_fc7, _b_fc7 = sess.run([w_fc6, b_fc6, w_fc7, b_fc7])


      mod_w_fc6 = np.zeros((3, 3, 512, 1024))
      mod_b_fc6 = np.zeros(1024)

      for i in range(1024):
        mod_b_fc6[i] = _b_fc6[4*i]
        for h in range(3):
          for w in range(3):
              mod_w_fc6[h, w, :, i] = _w_fc6[3*h, 3*w, :, 4*i]


      mod_w_fc7 = np.zeros((1, 1, 1024, 1024))
      mod_b_fc7 = np.zeros(1024)

      for i in range(1024):
        mod_b_fc7[i] = _b_fc7[4*i]
        for j in range(1024):
          mod_w_fc7[:, :, j, i] = _w_fc7[:, :, 4*j, 4*i]

      pre_weights["mod_w_fc6"] = mod_w_fc6
      pre_weights["mod_b_fc6"] = mod_b_fc6
      pre_weights["mod_w_fc7"] = mod_w_fc7
      pre_weights["mod_b_fc7"] = mod_b_fc7

  return net, end_points, init_flag, pre_weights
