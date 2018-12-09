"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from .modeler import Modeler


class ImageClassificationModeler(Modeler):
  def __init__(self, config, net):
    super(ImageClassificationModeler, self).__init__(config, net)

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    if self.config.mode == "train":
      self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_graph_fn(self, input):
    is_training = (self.config.mode == "train")
    return self.net(input,
                    self.config.num_classes,
                    is_training=is_training,
                    data_format=self.config.data_format)

  def create_eval_metrics_fn(self, predictions, labels):
    equality = tf.equal(predictions["classes"],
                        tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy

  def create_loss_fn(self, logits, labels):

    self.gether_train_vars()

    loss_cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

    loss_l2 = self.l2_regularization()

    loss = tf.identity(loss_cross_entropy + loss_l2, "total_loss")
    return loss

  def model_fn(self, x):
    images = x[0]
    labels = x[1]
    logits, predictions = self.create_graph_fn(images)

    if self.config.mode == "train":
      loss = self.create_loss_fn(logits, labels)
      grads = self.create_grad_fn(loss)
      accuracy = self.create_eval_metrics_fn(
        predictions, labels)
      return {"loss": loss,
              "grads": grads,
              "accuracy": accuracy,
              "learning_rate": self.learning_rate}
    elif self.config.mode == "eval":
      loss = self.create_loss_fn(logits, labels)
      accuracy = self.create_eval_metrics_fn(
        predictions, labels)
      return {"loss": loss,
              "accuracy": accuracy}
    elif self.config.mode == "infer":
      return {"classes": predictions["classes"],
              "probabilities": predictions["probabilities"]}


def build(config, network):
  return ImageClassificationModeler(config, network)
