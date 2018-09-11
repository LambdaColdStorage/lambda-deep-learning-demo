"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from modeler import Modeler


class TextGenerationTXTModeler(Modeler):
  def __init__(self, args):
    super(TextGenerationTXTModeler, self).__init__(args)

    if self.args.mode == "train":
      self.create_callbacks(["train_basic", "train_loss",
                             "train_accuracy", "train_speed",
                             "train_summary"])
    elif self.args.mode == "eval":
      self.create_callbacks(["eval_basic", "eval_loss",
                             "eval_accuracy", "eval_speed",
                             "eval_summary"])
    elif self.args.mode == "infer":
      self.create_callbacks(["infer_basic",
                             "infer_display_char_rnn"])

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_graph_fn(self, input):
    is_training = (self.args.mode == "train")
    return self.net(input, self.args.num_classes,
                    is_training=is_training, data_format=self.args.data_format)

  def create_eval_metrics_fn(self, predictions, labels):
    pass

  def create_loss_fn(self, logits, labels):
    pass

  def model_fn(self, x):
    pass


def build(args):
  return TextGenerationTXTModeler(args)
