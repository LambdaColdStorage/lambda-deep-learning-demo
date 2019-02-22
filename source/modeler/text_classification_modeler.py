"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from .modeler import Modeler
from source.optimizer import custom


class TextClassificationModeler(Modeler):
  def __init__(self, config, net):
    super(TextClassificationModeler, self).__init__(config, net)

    self.warmup_proportion = 0.1

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.vocab_size = inputter.get_vocab_size()
    self.embd = inputter.get_embd()
    self.epochs = inputter.get_num_epochs()

    if self.config.mode == "train":
      batch_size = (self.config.batch_size_per_gpu *
                    self.config.gpu_count)
      self.num_train_steps = int(
          self.num_samples / batch_size * float(self.epochs))
      self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    if self.config.mode == "train":
      self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_learning_rate_fn(self, global_step):
    if self.config.lr_method == "step":
      learning_rate = super(TextClassificationModeler, self).create_learning_rate_fn(global_step)
    else:
      learning_rate = tf.constant(value=self.config.learning_rate, shape=[], dtype=tf.float32)
      # Implements linear decay of the learning rate.
      learning_rate = tf.train.polynomial_decay(
          learning_rate,
          global_step,
          self.num_train_steps,
          end_learning_rate=0.0,
          power=1.0,
          cycle=False)

      if self.num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(self.num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = self.config.learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

      tf.identity(learning_rate, name="learning_rate") 
    return learning_rate

  def create_graph_fn(self, inputs, masks):
    is_training = (self.config.mode == "train")
    return self.net(inputs,
                    masks,
                    self.config.num_classes,
                    is_training,
                    self.config.batch_size_per_gpu,
                    self.vocab_size,
                    embd=self.embd,
                    use_one_hot_embeddings=False)

  def create_eval_metrics_fn(self, logits, labels):
    classes = tf.argmax(logits, axis=1, output_type=tf.int32)
    equality = tf.equal(classes, tf.reshape(labels, [-1]))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy

  def create_loss_fn(self, logits, labels):

      self.gether_train_vars()

      loss_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=tf.reshape(labels, [-1])))

      loss = tf.identity(loss_cross_entropy, "total_loss")

      return loss


  def create_optimizer(self, learning_rate):

    if self.config.optimizer == "custom":
      optimizer = custom.AdamWeightDecayOptimizer(
          learning_rate=learning_rate,
          weight_decay_rate=0.01,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])    
    else:
      optimizer = super(TextClassificationModeler, self).create_optimizer(learning_rate)

    return optimizer

  def create_grad_fn(self, loss, device_id=None, clipping=None):

    # Only update global step for the first GPU
    if device_id == 0 and self.config.optimizer == "custom":

      op_update_global_step = self.global_step.assign(self.global_step + 1)

      with tf.control_dependencies([op_update_global_step]):
        self.optimizer = self.create_optimizer(self.learning_rate)
        grads = self.optimizer.compute_gradients(loss, var_list=self.train_vars)
        if clipping:
          grads = [(tf.clip_by_value(g, -clipping, clipping), v) for g, v in grads]
    else:
        self.optimizer = self.create_optimizer(self.learning_rate)
        grads = self.optimizer.compute_gradients(loss, var_list=self.train_vars)
        if clipping:
          grads = [(tf.clip_by_value(g, -clipping, clipping), v) for g, v in grads]      

    return grads

  def model_fn(self, x, device_id=None):
    inputs = x[0]
    labels = x[1]
    masks = x[2]

    logits, probabilities = self.create_graph_fn(inputs, masks)

    if self.config.mode == "train":
      loss = self.create_loss_fn(logits, labels)
      grads = self.create_grad_fn(loss, device_id)
      accuracy = self.create_eval_metrics_fn(logits, labels)
      return {"loss": loss,
              "grads": grads,
              "accuracy": accuracy,
              "learning_rate": self.learning_rate}
    elif self.config.mode == "eval":
      loss = self.create_loss_fn(logits, labels)
      accuracy = self.create_eval_metrics_fn(
        logits, labels)
      return {"loss": loss,
              "accuracy": accuracy}
    elif self.config.mode == "infer":
      return {"classes": tf.argmax(logits, axis=1, output_type=tf.int32),
              "probabilities": probabilities}
    elif self.config.mode == "export":
      pass


def build(config, net):
  return TextClassificationModeler(config, net)
