"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
Train:
python demo/image_segmentation.py --mode=train \
--gpu_count=1 --batch_size_per_gpu=16 --epochs=200 \
--learning_rate=0.01 \
--piecewise_boundaries=50 \
--piecewise_lr_decay=1.0,0.1 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz \
--dataset_meta=~/demo/data/camvid/train.csv \
--model_dir=~/demo/model/image_segmentation_camvid

Evaluation:
python demo/image_segmentation.py --mode=eval \
--gpu_count=1 --batch_size_per_gpu=16 --epochs=1 \
--dataset_meta=~/demo/data/camvid/val.csv \
--model_dir=~/demo/model/image_segmentation_camvid

Infer:
python demo/image_segmentation.py --mode=infer \
--batch_size_per_gpu=1 --epochs=1 --gpu_count=1 \
--model_dir=~/demo/model/image_segmentation_camvid \
--test_samples=~/demo/data/camvid/test/0001TP_008550.png,~/demo/data/camvid/test/Seq05VD_f02760.png,~/demo/data/camvid/test/Seq05VD_f04650.png,~/demo/data/camvid/test/Seq05VD_f05100.png

Tune:
python demo/image_segmentation.py --mode=tune \
--batch_size_per_gpu=16 \
--dataset_meta=~/demo/data/camvid/train.csv \
--model_dir=~/demo/model/image_segmentation_camvid \
--gpu_count=1
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

  from source.config.image_segmentation_config import \
    ImageSegmentationCallbackConfig, ImageSegmentationInputterConfig, \
    ImageSegmentationModelerConfig

  parser = config_parser.default_parser()

  parser.add_argument("--augmenter",
                      type=str,
                      help="Name of the augmenter",
                      default="fcn_augmenter")
  parser.add_argument("--network", choices=["fcn"],
                      type=str,
                      help="Choose a network architecture",
                      default="fcn")
  parser.add_argument("--num_classes",
                      help="Number of classes.",
                      type=int,
                      default=12)
  parser.add_argument("--image_height",
                      help="Image height.",
                      type=int,
                      default=360)
  parser.add_argument("--image_width",
                      help="Image width.",
                      type=int,
                      default=480)
  parser.add_argument("--image_depth",
                      help="Number of color channels.",
                      type=int,
                      default=3)  
  parser.add_argument("--output_height",
                      help="Output height.",
                      type=int,
                      default=368)
  parser.add_argument("--output_width",
                      help="Output width.",
                      type=int,
                      default=480)
  parser.add_argument("--resize_side_min",
                      help="The minimal image size in augmentation.",
                      type=int,
                      default=400)
  parser.add_argument("--resize_side_max",
                      help="The maximul image size in augmentation.",
                      type=int,
                      default=600)
  parser.add_argument("--data_format",
                      help="channels_first or channels_last",
                      default="channels_first")
  parser.add_argument("--dataset_url",
                      help="URL for downloading data",
                      default="https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz")
  parser.add_argument("--train_callbacks",
                      help="List of callbacks in training.",
                      type=str,
                      default="train_basic,train_loss,train_accuracy,train_speed,train_summary")
  parser.add_argument("--eval_callbacks",
                      help="List of callbacks in evaluation.",
                      type=str,
                      default="eval_basic,eval_loss,eval_accuracy,eval_speed,eval_summary")
  parser.add_argument("--infer_callbacks",
                      help="List of callbacks in inference.",
                      type=str,
                      default="infer_basic,infer_display_image_segmentation")

  config = parser.parse_args()

  config = config_parser.prepare(config)

  # Download data if necessary
  if config.mode != "infer":
    if not os.path.exists(config.dataset_meta):
      downloader.download_and_extract(config.dataset_meta,
                                      config.dataset_url, False)
    else:
      print("Found " + config.dataset_meta + ".")

  # Generate config
  runner_config, callback_config, inputter_config, modeler_config = \
    config_parser.default_config(config)

  callback_config = ImageSegmentationCallbackConfig(
    callback_config,
    num_classes=config.num_classes)

  inputter_config = ImageSegmentationInputterConfig(
    inputter_config,
    image_height=config.image_height,
    image_width=config.image_width,
    image_depth=config.image_depth,
    output_height=config.output_height,
    output_width=config.output_width,
    resize_side_min=config.resize_side_min,
    resize_side_max=config.resize_side_max,                 
    num_classes=config.num_classes)

  modeler_config = ImageSegmentationModelerConfig(
    modeler_config,
    num_classes=config.num_classes,
    data_format=config.data_format)

  if config.mode == "tune":
    augmenter = (None if not config.augmenter else
                 importlib.import_module(
                  "source.augmenter." + config.augmenter))

    net = getattr(importlib.import_module(
      "source.network." + config.network), "net")

    inputter_module = importlib.import_module(
      "source.inputter.image_segmentation_csv_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.image_segmentation_modeler")
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
      "source.inputter.image_segmentation_csv_inputter").build(inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.image_segmentation_modeler").build(modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(runner_config, inputter, modeler, callbacks)

    # Run application
    runner.run()


if __name__ == "__main__":
  main()
