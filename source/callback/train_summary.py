"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from callback import Callback


class TrainSummary(Callback):
  def __init__(self, config):
    super(TrainSummary, self).__init__(config)

  def before_run(self, sess, saver):
    self.graph = tf.get_default_graph()

  def after_run(self, sess, saver, summary_writer):
    summary_writer.flush()
    summary_writer.close()

  def after_step(self, sess, outputs_dict, saver, summary_writer, feed_dict=None):

    global_step_op = self.graph.get_tensor_by_name("global_step:0")

    global_step = sess.run(global_step_op)

    if global_step % self.config.save_summary_steps == 0:
      summary_writer.add_summary(outputs_dict["summary"],
                                 global_step)


def build(config):
  return TrainSummary(config)
