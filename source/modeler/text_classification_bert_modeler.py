"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from .modeler import Modeler
from source.optimizer import custom


class TextClassificationBertModeler(Modeler):
  def __init__(self, config, net):
    super(TextClassificationBertModeler, self).__init__(config, net)

    self.bert_config = {
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "max_position_embeddings": 512,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "type_vocab_size": 2,
      "vocab_size": 30522
    }

    self.warmup_proportion = 0.1

  def get_dataset_info(self, inputter):
    self.num_samples = inputter.get_num_samples()
    self.num_classes = inputter.get_num_classes()
    self.epochs = inputter.get_num_epochs()

    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)

    if self.config.mode == "train":
      self.num_train_steps = int(
          self.num_samples / batch_size * float(self.epochs))
      self.num_warmup_steps = int(self.num_samples * self.warmup_proportion)      


  def create_nonreplicated_fn(self):
    self.global_step = tf.train.get_or_create_global_step()
    if self.config.mode == "train":
      self.learning_rate = self.create_learning_rate_fn(self.global_step)

  def create_graph_fn(self, is_training, input_ids, input_mask,
                      segment_ids, labels, num_classes, use_one_hot_embeddings):
    return self.net(self.bert_config,
                    is_training,
                    input_ids,
                    input_mask,
                    segment_ids,
                    labels,
                    num_classes,
                    use_one_hot_embeddings)

  def create_eval_metrics_fn(self, logits, labels):
    equality = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32),
                        labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32)) 
    tf.summary.scalar("accuracy", accuracy)  
    return accuracy

  def create_loss_fn(self, logits, labels):
    
    self.gether_train_vars()

    with tf.variable_scope("loss"):

      log_probs = tf.nn.log_softmax(logits, axis=-1)
      one_hot_labels = tf.one_hot(labels, depth=self.num_classes, dtype=tf.float32)

      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)
      tf.summary.scalar("total_loss", loss)

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
      optimizer = super(TextClassificationBertModeler, self).create_optimizer(learning_rate)

    return optimizer

  def create_learning_rate_fn(self, global_step):
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
    input_ids, input_mask, segment_ids, label_ids, is_real_example = x

    is_real_example = tf.cast(is_real_example, dtype=tf.float32)

    is_training = (self.config.mode == "train")

    logits, probabilities = self.create_graph_fn(is_training, input_ids,
                                                 input_mask, segment_ids, label_ids,
                                                 self.num_classes, False)

    if self.config.mode == "train":
      loss = self.create_loss_fn(logits, label_ids)
      grads = self.create_grad_fn(loss, device_id)
      accuracy = self.create_eval_metrics_fn(logits, label_ids)
      return {"loss": loss,
              "grads": grads,
              "accuracy": accuracy,
              "learning_rate": self.learning_rate}
    elif self.config.mode == "eval":
      loss = self.create_loss_fn(logits, label_ids)
      accuracy = self.create_eval_metrics_fn(logits, label_ids)
      return {"loss": loss,
              "accuracy": accuracy}
    elif self.config.mode == "infer":
      pass
    #   pass
    #   return {"classes": tf.argmax(logits, axis=1, output_type=tf.int32),
    #           "probabilities": probabilities}
    elif self.config.mode == "export":
      pass

def build(config, net):
  return TextClassificationBertModeler(config, net)
