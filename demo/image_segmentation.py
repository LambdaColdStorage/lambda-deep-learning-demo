"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
"""
import sys
import os
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

  config = parser.parse_args()

  config = config_parser.prepare(config)

  # Download data if necessary
  if config.mode != "infer":
    if hasattr(config, "dataset_meta"):
      if not os.path.exists(config.dataset_meta):
        downloader.download_and_extract(config.dataset_meta,
                                        config.dataset_url,
                                        False)
      else:
        print("Found " + config.dataset_meta + ".")
    elif hasattr(config, "train_dataset_meta"):
      if not os.path.exists(config.train_dataset_meta):
        print(config.train_dataset_meta)
        downloader.download_and_extract(config.train_dataset_meta,
                                        config.dataset_url,
                                        False)
      else:
        print("Found " + config.train_dataset_meta + ".")
    else:
      assert False, "A meta data must be provided."

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

    callbacks = []
    for name in config.callbacks:
      callback = importlib.import_module(
        "source.callback." + name).build(callback_config)
      callbacks.append(callback)

    inputter = importlib.import_module(
      "source.inputter.image_segmentation_csv_inputter").build(
      inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.image_segmentation_modeler").build(
      modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(
      runner_config, inputter, modeler, callbacks)

    # Run application
    runner.run()


if __name__ == "__main__":
  main()
