import numpy as np

import tensorflow as tf

from source.network.bert import bert
from source.network.bert import bert_common

# BERT basic
bert_config = {
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


def net(input_ids, input_mask, num_labels, is_training, batch_size, vocab_size, embd=None, use_one_hot_embeddings=False):

  segment_ids = tf.zeros_like(input_ids)

  model = bert.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope="bert")

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  with tf.variable_scope("classification", reuse=tf.AUTO_REUSE):
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

  if is_training:
    # I.e., 0.1 dropout
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

  logits = tf.matmul(output_layer, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)
  probabilities = tf.nn.softmax(logits, axis=-1)

  return logits, probabilities