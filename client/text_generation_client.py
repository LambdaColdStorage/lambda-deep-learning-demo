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


python client/text_generation_client.py --unit=word --starter=218 --length=128

saved_model_cli show --dir ~/demo/model/char_rnn_shakespeare/export/1/ --all

"""

from __future__ import print_function

import requests
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

def main():

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--unit", choices=["char", "word"],
                      type=str,
                      help="Choose a job mode from char and word.",
                      default="char")

  parser.add_argument("--length",
                      type=int,
                      help="Length (in number of units) of generated text.",
                      default=128)

  parser.add_argument("--starter",
                      type=int,
                      help="id of the starting item. For example, 218 is Duke for word_rnn, 28 is T for char_rnn",
                      default=8)

  parser.add_argument("-rnn_size",
                      type=int,
                      help="Size of RNN. Has to match the served model",
                      default=256)

  args = parser.parse_args()

  input_item = np.full((1, 1), args.starter, dtype=np.int32)

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
    input_dict["c0"] = c0
    input_dict["h0"] = h0
    input_dict["c1"] = c1
    input_dict["h1"] = h1
    data = json.dumps({"signature_name": "predict", "instances": [input_dict]})


    headers = {"content-type": "application/json"}

    response = requests.post(SERVER_URL, data=data, headers=headers)
    # print(response)
    # response.raise_for_status()
    predictions = response.json()["predictions"][0]
    
    items = predictions["items"]
    for p in predictions["output_probabilities"]:
      pick_id = pick(p)
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
    c1 = predictions["output_last_state"][1][1][0]
    h1 = predictions["output_last_state"][1][1][0]

  results = items[args.starter] + " " + results
  print(results)

if __name__ == '__main__':
  main()
