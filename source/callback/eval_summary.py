"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import os

import tensorflow as tf

from .callback import Callback


class EvalSummary(Callback):
  def __init__(self, config):
    super(EvalSummary, self).__init__(config)

  def before_run(self, sess):
    self.graph = tf.get_default_graph()

    # Create summary writer
    if self.config.mode == "train":
      self.summary_writer = tf.summary.FileWriter(
        self.config.model_dir,
        graph=self.graph)
    elif self.config.mode == "eval":
      self.summary_writer = tf.summary.FileWriter(
        os.path.join(self.config.model_dir, "eval"),
        graph=self.graph)

    self.accumulated_summary = {}
    self.global_step = 0
    global_step_op = self.graph.get_tensor_by_name("global_step:0")
    self.trained_step = sess.run(global_step_op)

  def after_run(self, sess):
    summary = tf.Summary()
    for key in self.accumulated_summary:
      summary.value.add(tag=key,
                        simple_value=(self.accumulated_summary[key] /
                                      self.global_step))
    self.summary_writer.add_summary(summary, self.trained_step)
    self.summary_writer.flush()
    self.summary_writer.close()

  def after_step(self, sess, outputs_dict, feed_dict=None):
    if not self.accumulated_summary:
      self.accumulated_summary = outputs_dict
    else:
      for key in outputs_dict:
        self.accumulated_summary[key] = (
          self.accumulated_summary[key] + outputs_dict[key])
    self.global_step = self.global_step + 1


def build(config):
  return EvalSummary(config)
