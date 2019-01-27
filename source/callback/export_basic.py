"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from __future__ import print_function

import os
import sys
import glob
import shutil

import tensorflow as tf

from .callback import Callback


class ExportBasic(Callback):
  def __init__(self, config):
    super(ExportBasic, self).__init__(config)

  def before_run(self, sess):

    export_path = os.path.join(self.config.model_dir,
                               self.config.export_dir,
                               self.config.export_version)
    if os.path.isdir(export_path):
      shutil.rmtree(export_path)

    self.graph = tf.get_default_graph()

    # Create saver
    self.saver = tf.train.Saver(name="global_saver")

    ckpt_path = os.path.join(self.config.model_dir, "*ckpt*")
    if os.path.isdir(self.config.model_dir):
      if tf.train.checkpoint_exists(ckpt_path):
        self.saver.restore(sess,
                           tf.train.latest_checkpoint(self.config.model_dir))
        print("Parameters restored.")
      else:
        sess.run(tf.global_variables_initializer())
        print("Can not find checkpoint at " + ckpt_path + ", use default initialization.")
    else:
      print("Can not find checkpoint at " + ckpt_path + ", use default initialization.")
      sess.run(tf.global_variables_initializer())

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    dict_inputs = {}
    dict_outputs = {}
    for name_input_ops in self.config.input_ops:
      dict_inputs[name_input_ops] = tf.saved_model.utils.build_tensor_info(
        self.graph.get_tensor_by_name(name_input_ops + ":0"))
    for name_output_ops in self.config.output_ops:
      dict_outputs[name_output_ops] = tf.saved_model.utils.build_tensor_info(
        self.graph.get_tensor_by_name(name_output_ops + ":0"))

    predict_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs=dict_inputs,
          outputs=dict_outputs,
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={'predict':predict_signature},
      main_op=tf.tables_initializer(),
      strip_default_attrs=True)

    builder.save()


  def after_run(self, sess):
    pass


  def after_step(self, sess, outputs_dict, feed_dict=None):
    pass


def build(config):
  return ExportBasic(config)
