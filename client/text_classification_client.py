# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A client that performs inferences on a ResNet model using the REST API.

Install nvidia-docker
https://github.com/NVIDIA/nvidia-docker#quick-start

Typical usage example:
docker run --runtime=nvidia -p 8501:8501 \
--name tfserving_textclassification \
--mount type=bind,source=/home/ubuntu/demo/model/seq2label_basic_Imdb/export,target=/models/textclassification \
-e MODEL_NAME=textclassification -t tensorflow/serving:latest-gpu &

docker run --runtime=nvidia -p 8501:8501 \
--name tfserving_textclassification \
--mount type=bind,source=/home/ubuntu/demo/model/seq2label_glove_Imdb/export,target=/models/textclassification \
-e MODEL_NAME=textclassification -t tensorflow/serving:latest-gpu &

docker run --runtime=nvidia -p 8501:8501 \
--name tfserving_textclassification \
--mount type=bind,source=/home/ubuntu/demo/model/seq2label_bert_Imdb/export,target=/models/textclassification \
-e MODEL_NAME=textclassification -t tensorflow/serving:latest-gpu &


python client/text_classification_client.py \
  --vocab_file=/home/ubuntu/demo/data/IMDB/vocab_basic.txt \
  --vocab_top_k=40000 \
  --encode_method=basic


python client/text_classification_client.py \
  --vocab_file=/home/ubuntu/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_top_k=40000 \
  --encode_method=basic

python client/text_classification_client.py \
  --vocab_file=/home/ubuntu/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_top_k=-1 \
  --encode_method=bert

saved_model_cli show --dir ~/demo/model/textclassification/export/1/ --all

"""

from __future__ import print_function

import os
import sys
import pickle
import requests
import numpy as np
import json
import argparse
import re


# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/textclassification:predict'


def basic_encode(sentences, vocab, max_seq_length):

  def run(sentence):
    encode_sentence = [vocab[w] for w in sentence if w in vocab]
    if max_seq_length > 0:
      encode_sentence = encode_sentence[0:max_seq_length]

    mask = [1] * len(encode_sentence)

    if max_seq_length > 0:
      while len(encode_sentence) < max_seq_length:
        encode_sentence.append(0)
        mask.append(0)

    return np.array(encode_sentence, dtype="int32"), np.array(mask, dtype="int32")

  encode_sentences, encode_masks = zip(*[run (s) for s in sentences])
  return encode_sentences, encode_masks

def bert_encode(sentences, vocab, max_seq_length):

  def run(sentence):

    # Add special tokens to the sentence
    if len(sentence) > max_seq_length - 2:
      sentence = sentence[0:(max_seq_length - 2)]
    tokens = []
    tokens.append("[CLS]")
    for token in sentence:
      tokens.append(token)
    tokens.append("[SEP]")

    # Encode sentence by vocabulary id
    encode_sentence = [vocab[w] for w in tokens if w in vocab]

    mask = [1] * len(encode_sentence)
    while len(encode_sentence) < max_seq_length:
      encode_sentence.append(0)
      mask.append(0)

    return np.array(encode_sentence, dtype="int64"), np.array(mask, dtype="int64")

  encode_sentences, masks = zip(*[run (s) for s in sentences])
  return encode_sentences, masks

def main():

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--max_length",
                      type=int,
                      help="Max length (in number of characters) of the input text.",
                      default=256)
  parser.add_argument("--input_text",
                      type=str,
                      help="Input text for classification",
                      default="This movie is awesome.#This movie is bad.")
  parser.add_argument("--vocab_file",
                      help="Path of the vocabulary file.",
                      type=str,
                      default="")
  parser.add_argument("--encode_method",
                      help="Name of the method to encode text.",
                      type=str,
                      default="basic")
  parser.add_argument("--vocab_top_k",
                      help="Number of words kept in the vocab. set to -1 to use all words.",
                      type=int,
                      default=-1)
  parser.add_argument("--vocab_format",
                      help="Format of vocabulary.",
                      type=str,
                      default="pickle",
                      choices=["pickle", "txt"])  
  parser.add_argument("--splitter",
                      help="A special character to split test_samples into a list",
                      type=str,
                      default="#")

  args = parser.parse_args()

  args.vocab_file = os.path.expanduser(args.vocab_file)

  sys.path.append('.')
  from demo.text.preprocess import vocab_loader

  vocab, items, embd = vocab_loader.load(args.vocab_file, args.vocab_format, args.vocab_top_k)

  list_input_text = args.input_text.split(args.splitter)
  sentences = []
  for s in list_input_text:
    sentences.append(re.findall(r"[\w']+|[.,!?;]", s))

  if args.encode_method == "basic":
    encode_sentences, encode_masks = basic_encode(sentences, vocab, args.max_length)
  elif args.encode_method == "bert":
    encode_sentences, encode_masks = bert_encode(sentences, vocab, args.max_length)

  for es, m, s in zip(encode_sentences, encode_masks, list_input_text):
    input_text = es.tolist()
    input_mask = m.tolist()

    input_dict = {}
    input_dict["input_text"] = input_text
    input_dict["input_mask"] = input_mask

    data = json.dumps({"signature_name": "predict", "instances": [input_dict]})

    headers = {"content-type": "application/json"}

    response = requests.post(SERVER_URL, data=data, headers=headers)

    p = response.json()["predictions"][0][0]
    probability = max(p)
    label = p.index(probability)
    print(s)
    print("class: " + str(label) + ", prob: " + str(probability))

if __name__ == '__main__':
  main()
