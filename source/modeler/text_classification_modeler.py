"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from .modeler import Modeler

class TextClassificationModeler(Modeler):
  def __init__(self, config, net):
    super(TextClassificationModeler, self).__init__(config, net)

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.vocab_size = inputter.get_vocab_size()

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    if self.config.mode == "train":
      self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_graph_fn(self, inputs):
    pass

  def create_eval_metrics_fn(self, logits, labels):
    pass

  def create_loss_fn(self, logits, labels):
    pass

  def model_fn(self, x, device_id=None):
    pass


def build(config, net):
  return TextClassificationModeler(config, net)
