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
--name tfserving_textgeneration \
--mount type=bind,source=/home/ubuntu/demo/model/char_rnn_shakespeare/export,target=/models/textgeneration \
-e MODEL_NAME=textgeneration -t tensorflow/serving:latest-gpu &


docker run --runtime=nvidia -p 8501:8501 \
--name tfserving_textgeneration \
--mount type=bind,source=/home/ubuntu/demo/model/word_rnn_shakespeare/export,target=/models/textgeneration \
-e MODEL_NAME=textgeneration -t tensorflow/serving:latest-gpu &


docker run --runtime=nvidia -p 8501:8501 \
--name tfserving_textgeneration \
--mount type=bind,source=/home/ubuntu/demo/model/word_rnn_glove_shakespeare/export,target=/models/textgeneration \
-e MODEL_NAME=textgeneration -t tensorflow/serving:latest-gpu &


python client/text_generation_client.py \
--vocab_file=~/demo/data/shakespeare/shakespeare_char_basic.vocab \
--vocab_top_k=-1 \
--vocab_format=pickle \
--unit=char --starter=T --length=256

saved_model_cli show --dir ~/demo/model/char_rnn_shakespeare/export/1/ --all

"""

from __future__ import print_function

import requests
import os
import pickle
import numpy as np
import json
import argparse


# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/textgeneration:predict'

def pick(prob):
  t = np.cumsum(prob)
  s = np.sum(prob)
  return(int(np.searchsorted(t, np.random.rand(1) * s)))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def loadVocab(vocab_file, vocab_format, top_k):
  if vocab_format == "pickle":
    f = open(vocab_file,'r')  
    items = pickle.load(f)
    if top_k > 0 and len(items) > top_k:
      items = items[:top_k]
    vocab = { w : i for i, w in enumerate(items)}
    embd = None
  elif vocab_format == "txt":
    items = []
    embd = []
    file = open(vocab_file,'r')
    count = 0
    for line in file.readlines():
        row = line.strip().split(' ')
        items.append(row[0])
        
        if len(row) > 1:
          embd.append(row[1:])

        count += 1
        if count == top_k:
          break
    file.close()
    vocab = { w : i for i, w in enumerate(items)}
    if embd:
      embd = np.asarray(embd).astype(np.float32)

  return vocab, items, embd


def main():

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--vocab_file",
                      help="Path of the vocabulary file.",
                      type=str,
                      default="")

  parser.add_argument("--vocab_top_k",
                      help="Number of words kept in the vocab. set to -1 to use all words.",
                      type=int,
                      default=-1)

  parser.add_argument("--vocab_format",
                      help="Format of vocabulary.",
                      type=str,
                      default="pickle",
                      choices=["pickle", "txt"])

  parser.add_argument("--unit", choices=["char", "word"],
                      type=str,
                      help="Choose a job mode from char and word.",
                      default="char")

  parser.add_argument("--length",
                      type=int,
                      help="Length (in number of units) of generated text.",
                      default=128)

  parser.add_argument("--softmax_temperature",
                      help="Control the randomness during generation.",
                      type=float,
                      default=1.0)

  parser.add_argument("--starter",
                      type=str,
                      help="id of the starting item. For example, 218 is Duke for word_rnn, 28 is T for char_rnn",
                      default="T")

  parser.add_argument("-rnn_size",
                      type=int,
                      help="Size of RNN. Has to match the served model",
                      default=256)

  args = parser.parse_args()


  args.vocab_file = os.path.expanduser(args.vocab_file)

  vocab, items, embd = loadVocab(
    args.vocab_file, args.vocab_format, args.vocab_top_k)

  input_item = np.full((1, 1), vocab[args.starter], dtype=np.int32)

  c0 = np.zeros((1, args.rnn_size), dtype=np.float32)
  h0 = np.zeros((1, args.rnn_size), dtype=np.float32)
  c1 = np.zeros((1, args.rnn_size), dtype=np.float32)
  h1 = np.zeros((1, args.rnn_size), dtype=np.float32)

  input_item = input_item.tolist()[0]
  c0 = c0.tolist()[0]
  h0 = h0.tolist()[0]
  c1 = c1.tolist()[0]
  h1 = h1.tolist()[0]

  results = ""

  for i_step in range(args.length):
    input_dict = {}
    input_dict["input_item"] = input_item
    input_dict["RNN/c0"] = c0
    input_dict["RNN/h0"] = h0
    input_dict["RNN/c1"] = c1
    input_dict["RNN/h1"] = h1
    data = json.dumps({"signature_name": "predict", "instances": [input_dict]})


    headers = {"content-type": "application/json"}

    response = requests.post(SERVER_URL, data=data, headers=headers)

    predictions = response.json()["predictions"][0]
    
    for p in predictions["output_logits"]:
      pick_id = pick(softmax( np.asarray(p, dtype=np.float32) / args.softmax_temperature))
      if args.unit == "char":
        results += items[pick_id]
      elif args.unit == "word":
        if items[pick_id] != "\n":
          results += items[pick_id] + " "
        else:
          results += items[pick_id]

    input_item = np.full((1, 1), pick_id, dtype=np.int32).tolist()[0]
    c0 = predictions["output_last_state"][0][0][0]
    h0 = predictions["output_last_state"][0][1][0]
    c1 = predictions["output_last_state"][1][0][0]
    h1 = predictions["output_last_state"][1][1][0]

  if args.unit == "char":
    results = args.starter + results
  elif args.unit == "word":
    results = args.starter + " " + results

  print('=======================================')
  print(results)
  print('=======================================')

if __name__ == '__main__':
  main()
