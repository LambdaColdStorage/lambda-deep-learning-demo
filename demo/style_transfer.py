"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
Train:
python demo/style_transfer.py --mode=train \
--gpu_count=1 --batch_size_per_gpu=4 --epochs=10 \
--piecewise_boundaries=5 \
--piecewise_lr_decay=1.0,0.1 \
--optimizer=rmsprop \
--learning_rate=0.01 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/mscoco_fns.tar.gz \
--dataset_meta=~/demo/data/mscoco_fns/train2014.csv \
--model_dir=~/demo/model/style_transfer_mscoco_fns \
--summary_names=loss,learning_rate \
--skip_trainable_vars=vgg_19

Eval:
python demo/style_transfer.py --mode=eval \
--gpu_count=1 --batch_size_per_gpu=4 --epochs=1 \
--dataset_meta=~/demo/data/mscoco_fns/eval2014.csv \
--model_dir=~/demo/model/style_transfer_mscoco_fns

Infer:
python demo/style_transfer.py --mode=infer \
--batch_size_per_gpu=1 --epochs=1 --gpu_count=1 \
--model_dir=~/demo/model/style_transfer_mscoco_fns \
--test_samples=~/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg

Tune:
python demo/style_transfer.py --mode=tune \
--gpu_count=1 --batch_size_per_gpu=4 \
--dataset_meta=~/demo/data/mscoco_fns/train2014.csv \
--model_dir=~/demo/model/style_transfer_mscoco_fns \
--summary_names=loss,learning_rate \
--skip_trainable_vars=vgg_19
"""
import sys
import os
import argparse
import importlib


def main():

  sys.path.append('.')

  from source.tool import downloader
  from source.tool import tuner
  from source.tool import config_parser

  from source.config.style_transfer_config import \
    StyleTransferCallbackConfig, StyleTransferInputterConfig, \
    StyleTransferModelerConfig
  

  parser = config_parser.default_parser()

  parser.add_argument("--augmenter",
                      type=str,
                      help="Name of the augmenter",
                      default="fns_augmenter")
  parser.add_argument("--network", choices=["fns"],
                      type=str,
                      help="Choose a network architecture",
                      default="fns")
  parser.add_argument("--style_weight",
                      help="Weight for style loss",
                      default=100)
  parser.add_argument("--content_weight",
                      help="Weight for content loss",
                      default=7.5)
  parser.add_argument("--tv_weight",
                      help="Weight for tv loss",
                      default=200)
  parser.add_argument("--image_height",
                      help="Image height.",
                      type=int,
                      default=256)
  parser.add_argument("--image_width",
                      help="Image width.",
                      type=int,
                      default=256)
  parser.add_argument("--resize_side_min",
                      help="The minimal image size in augmentation.",
                      type=int,
                      default=400)
  parser.add_argument("--resize_side_max",
                      help="The maximul image size in augmentation.",
                      type=int,
                      default=600)
  parser.add_argument("--image_depth",
                      help="Number of color channels.",
                      type=int,
                      default=3)
  parser.add_argument("--feature_net",
                      help="Name of feature net",
                      default="vgg_19_conv")
  parser.add_argument("--feature_net_path",
                      help="Path to pre-trained vgg model.",
                      default=os.path.join(
                        os.environ['HOME'],
                        "demo/model/vgg_19_2016_08_28/vgg_19.ckpt"))
  parser.add_argument("--style_image_path",
                      help="Path to style image",
                      default=os.path.join(os.environ['HOME'],
                                           "demo/data/mscoco_fns/gothic.jpg"))

  parser.add_argument("--data_format",
                      help="channels_first or channels_last",
                      choices=["channels_first", "channels_last"],
                      default="channels_last")
  parser.add_argument("--dataset_url",
                      help="URL for downloading data",
                      default="https://s3-us-west-2.amazonaws.com/lambdalabs-files/mscoco_fns.tar.gz")
  parser.add_argument("--feature_net_url",
                      help="URL for downloading pre-trained feature_net",
                      default="http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz")
  parser.add_argument("--train_callbacks",
                      help="List of callbacks in training.",
                      type=str,
                      default="train_basic,train_loss,train_speed,train_summary")
  parser.add_argument("--eval_callbacks",
                      help="List of callbacks in evaluation.",
                      type=str,
                      default="eval_basic,eval_loss,eval_speed,eval_summary")
  parser.add_argument("--infer_callbacks",
                      help="List of callbacks in inference.",
                      type=str,
                      default="infer_basic,infer_display_style_transfer")  


  config = parser.parse_args()

  config = config_parser.prepare(config)

  # Download data if necessary
  if config.mode != "infer":
    if not os.path.exists(config.dataset_meta):
      downloader.download_and_extract(config.dataset_meta,
                                      config.dataset_url, False)
    else:
      print("Found " + config.dataset_meta + ".")

    if not os.path.exists(config.feature_net_path):
      downloader.download_and_extract(config.feature_net_path,
                                      config.feature_net_url, True)
    else:
      print("Found " + config.feature_net_path + '.')

  # Generate config
  runner_config, callback_config, inputter_config, modeler_config = \
    config_parser.default_config(config)

  inputter_config = StyleTransferInputterConfig(
    inputter_config,
    image_height=config.image_height,
    image_width=config.image_width,
    image_depth=config.image_depth,
    resize_side_min = config.resize_side_min,
    resize_side_max = config.resize_side_max)

  modeler_config = StyleTransferModelerConfig(
    modeler_config,
    data_format = config.data_format,
    image_depth=config.image_depth,
    style_weight = config.style_weight,
    content_weight = config.content_weight,
    tv_weight = config.tv_weight,
    feature_net = config.feature_net,
    feature_net_path = config.feature_net_path,
    style_image_path = config.style_image_path)

  if config.mode == "tune":
    augmenter = (None if not config.augmenter else
                 importlib.import_module(
                  "source.augmenter." + config.augmenter))

    net = getattr(importlib.import_module(
      "source.network." + config.network), "net")

    inputter_module = importlib.import_module(
      "source.inputter.style_transfer_csv_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.style_transfer_modeler")
    runner_module = importlib.import_module(
      "source.runner.parameter_server_runner")

    tuner.tune(config,
               runner_config,
               callback_config,
               inputter_config,
               modeler_config,
               inputter_module,
               modeler_module,
               runner_module)
  else:

    """
    An application owns a runner.
    Runner: Distributes a job across devices, schedules the excution.
            It owns an inputter and a modeler.
    Inputter: Handles the data pipeline.
              It (optionally) owns a data augmenter.
    Modeler: Creates functions for network, loss, optimization and evaluation.
             It owns a network and a list of callbacks as inputs.
    """

    augmenter = (None if not config.augmenter else
                 importlib.import_module(
                  "source.augmenter." + config.augmenter))

    net = getattr(importlib.import_module(
      "source.network." + config.network), "net")

    if config.mode == "train":
      callback_names = config.train_callbacks
    elif config.mode == "eval":
      callback_names = config.eval_callbacks
    elif config.mode == "infer":
      callback_names = config.infer_callbacks

    callbacks = []
    for name in callback_names:
      callback = importlib.import_module(
        "source.callback." + name).build(callback_config)
      callbacks.append(callback)

    inputter = importlib.import_module(
      "source.inputter.style_transfer_csv_inputter").build(inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.style_transfer_modeler").build(modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(runner_config, inputter, modeler, callbacks)

    # Run application
    runner.run()

if __name__ == "__main__":
  main()
