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

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model

The SavedModel must be one that can take JPEG images as inputs.

Typical usage example:

docker run -p 8501:8501 --name tfserving_cifar --mount type=bind,source=/home/chuan/demo/model/cifar10-resnet32-20180824/export,target=/models/cifar -e MODEL_NAME=cifar -t tensorflow/serving &

    resnet_client.py
"""

from __future__ import print_function

import base64
import requests
import skimage.io
from skimage.transform import resize
import numpy as np
import json
import matplotlib.pyplot as plt

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/cifar:predict'

# The image URL is the location of the image we should send to the server
# IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
IMAGE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/1.png'


def main():
  # Download the image
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()

  # Compose a JSON Predict request (send JPEG image in base64).
  jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')

  imgdata = base64.b64decode(jpeg_bytes)
  image = skimage.io.imread(imgdata, plugin='imageio')

  data = json.dumps({"signature_name": "predict", "instances": image.tolist()})
  headers = {"content-type": "application/json"}

  response = requests.post(SERVER_URL, data=data, headers=headers)
  response.raise_for_status()
  print(response.json())

if __name__ == '__main__':
  main()
