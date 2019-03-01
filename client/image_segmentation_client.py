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
--name tfserving_segmentation \
--mount type=bind,source=/home/ubuntu/demo/model/fcn_camvid_20190125/export,target=/models/segmentation \
-e MODEL_NAME=segmentation -t tensorflow/serving:latest-gpu &


docker run --runtime=nvidia -p 8501:8501 \
--name tfserving_segmentation \
--mount type=bind,source=/home/ubuntu/demo/model/unet_camvid_20190125/export,target=/models/segmentation \
-e MODEL_NAME=segmentation -t tensorflow/serving:latest-gpu &

python client/image_segmentation_client.py
"""

from __future__ import print_function

import requests
import skimage.io
from skimage.transform import resize
from skimage import img_as_ubyte
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/segmentation:predict'

# The image URL is the location of the image we should send to the server
IMAGE_PATH = '/home/ubuntu/demo/data/camvid/test/0001TP_008550.png'


NUM_CLASSES = 12

COLORS = np.random.randint(255, size=(NUM_CLASSES, 3))

def render_label(label, num_classes, label_colors):

  label = label.astype(int)
  r = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
  g = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
  b = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)

  for i_color in range(0, num_classes):
    r[label == i_color] = label_colors[i_color, 0]
    g[label == i_color] = label_colors[i_color, 1]
    b[label == i_color] = label_colors[i_color, 2]

  rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
  rgb[:, :, 0] = r
  rgb[:, :, 1] = g
  rgb[:, :, 2] = b

  return rgb

def main():
  # Read the image
  image = img_as_ubyte(skimage.io.imread(IMAGE_PATH, plugin='imageio'))
  
  data = json.dumps({"signature_name": "predict", "instances": image.tolist()})
  headers = {"content-type": "application/json"}

  response = requests.post(SERVER_URL, data=data, headers=headers)
  response.raise_for_status()

  predictions = np.squeeze(
    np.array(response.json()["predictions"]), axis=0)
  
  render_image = render_label(predictions, NUM_CLASSES, COLORS)

  render_image = Image.fromarray(render_image, 'RGB')
  plt.imshow(render_image)
  plt.show()


if __name__ == '__main__':
  main()
