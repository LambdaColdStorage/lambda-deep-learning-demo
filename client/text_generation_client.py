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


python client/text_generation_client.py
"""

from __future__ import print_function

import requests
import numpy as np
import json


# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/textgeneration:predict'

START_CHAR = 28
NUM_CHAR = 1000
RNN_SIZE = 256

def main():
  input_chars = np.full((1, 1), START_CHAR, dtype=np.int32)
  # input_chars = np.array([[START_CHAR]], dtype=np.int32)
  # c0 = np.zeros((1, RNN_SIZE), dtype=np.float32)
  # h0 = np.zeros((1, RNN_SIZE), dtype=np.float32)
  # c1 = np.zeros((1, RNN_SIZE), dtype=np.float32)
  # h1 = np.zeros((1, RNN_SIZE), dtype=np.float32)

  c0 = np.zeros((1, RNN_SIZE), dtype=np.float32)
  h0 = np.zeros((1, RNN_SIZE), dtype=np.float32)
  c1 = np.zeros((1, RNN_SIZE), dtype=np.float32)
  h1 = np.zeros((1, RNN_SIZE), dtype=np.float32)

  # # print(input_chars.shape)
  # # print(c0.shape)
  # # print(h0.shape)
  # # print(c1.shape)
  # # print(h1.shape)

  input_chars = input_chars.tolist()[0]
  c0 = c0.tolist()[0]
  h0 = h0.tolist()[0]
  c1 = c1.tolist()[0]
  h1 = h1.tolist()[0]

  input_dict = {}
  input_dict["input_chars"] = input_chars
  input_dict["c0"] = c0
  input_dict["h0"] = h0
  input_dict["c1"] = c1
  input_dict["h1"] = h1
  data = json.dumps({"signature_name": "predict", "instances": [input_dict]})

  # print(data)

  headers = {"content-type": "application/json"}

  response = requests.post(SERVER_URL, data=data, headers=headers)
  # print(response.text)
  response.raise_for_status()

  predictions = response.json()["predictions"]

  print(predictions)
  print(type(predictions))
  
  # render_image = Image.fromarray(img_as_ubyte(predictions / 255.0), 'RGB')
  # plt.imshow(render_image)
  # plt.show()


if __name__ == '__main__':
  main()
