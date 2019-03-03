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
--name tfserving_objectdetection \
--mount type=bind,source=/home/ubuntu/demo/model/ssd300_mscoco_20190105/export,target=/models/objectdetection \
-e MODEL_NAME=objectdetection -t tensorflow/serving:latest-gpu &

python client/object_detection_client.py --image_path=/home/ubuntu/demo/data/mscoco_fns/val2014/COCO_val2014_000000301397.jpg


docker run --runtime=nvidia -p 8501:8501 \
--name tfserving_objectdetection \
--mount type=bind,source=/home/ubuntu/demo/model/ssd512_mscoco_20190105/export,target=/models/objectdetection \
-e MODEL_NAME=objectdetection -t tensorflow/serving:latest-gpu &

python client/object_detection_client.py --image_path=/home/ubuntu/demo/data/mscoco_fns/val2014/COCO_val2014_000000301397.jpg

"""

from __future__ import print_function

import os
import requests
import numpy as np
import json
import argparse
import skimage.io
from skimage.transform import resize
from skimage.transform import rescale
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from PIL import Image

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/objectdetection:predict'


MSCOCO_CAT_NAME = [u'person', u'bicycle', u'car', u'motorcycle', u'airplane',
                   u'bus', u'train', u'truck', u'boat', u'traffic light',
                   u'fire hydrant', u'stop sign', u'parking meter', u'bench',
                   u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow',
                   u'elephant', u'bear', u'zebra', u'giraffe', u'backpack',
                   u'umbrella', u'handbag', u'tie', u'suitcase', u'frisbee',
                   u'skis', u'snowboard', u'sports ball', u'kite',
                   u'baseball bat', u'baseball glove', u'skateboard',
                   u'surfboard', u'tennis racket', u'bottle', u'wine glass',
                   u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana',
                   u'apple', u'sandwich', u'orange', u'broccoli', u'carrot',
                   u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch',
                   u'potted plant', u'bed', u'dining table', u'toilet', u'tv',
                   u'laptop', u'mouse', u'remote', u'keyboard', u'cell phone',
                   u'microwave', u'oven', u'toaster', u'sink', u'refrigerator',
                   u'book', u'clock', u'vase', u'scissors', u'teddy bear',
                   u'hair drier', u'toothbrush']
COLORS = plt.cm.hsv(np.linspace(0, 1, 121)).tolist()

THRESHOLD = 0.5

def display_ori(image, outputs_dict):
  s = outputs_dict["output_scores"]
  l = outputs_dict["output_labels"]
  b = outputs_dict["output_bboxes"]

  h, w = image.shape[:2]
    
  plt.imshow(image)
  currentAxis = plt.gca()

  for score, label, box in zip(s, l, b):
    if score > THRESHOLD:
      box = np.asarray(box) * np.asarray([float(w), float(h), float(w), float(h)])
      xmin = np.clip(box[0], 0, w)
      ymin = np.clip(box[1], 0, h)
      xmax = np.clip(box[2], 0, w)
      ymax = np.clip(box[3], 0, h)
      label_name = MSCOCO_CAT_NAME[label - 1]

      display_txt = '%s: %.2f'%(label_name, score)

      coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
      color = COLORS[label]

      currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
      currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

  plt.axis('off')
  plt.show()

def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--image_path",
                      help="path for image to run inference",
                      default="~/demo/data/mscoco_fns/val2014/COCO_val2014_000000301397.jpg")

  args = parser.parse_args()

  args.image_path = os.path.expanduser(args.image_path)

  # Read the image
  image = skimage.io.imread(args.image_path, plugin='imageio')
  image = rescale(image, 2.0, anti_aliasing=False)
  image = img_as_ubyte(image)

  data = json.dumps({"signature_name": "predict", "instances": image.tolist()})
  headers = {"content-type": "application/json"}

  response = requests.post(SERVER_URL, data=data, headers=headers)
  response.raise_for_status()
  predictions = response.json()["predictions"]

  display_ori(image, predictions[0])

if __name__ == '__main__':
  main()
