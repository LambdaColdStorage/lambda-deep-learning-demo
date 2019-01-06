"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
"""
import sys
import os
import importlib

"""
Image Segmentation Demo
"""


def main():

  sys.path.append('.')

  from source.tool import downloader
  from source.tool import tuner
  from source.tool import config_parser

  from source.config.image_segmentation_config import \
      ImageSegmentationCallbackConfig, ImageSegmentationInputterConfig, \
      ImageSegmentationModelerConfig

  parser = config_parser.default_parser()

  app_parser = parser.add_argument_group('app')

  app_parser.add_argument("--num_classes",
                          help="Number of classes.",
                          type=int,
                          default=12)
  app_parser.add_argument("--image_height",
                          help="Image height.",
                          type=int,
                          default=360)
  app_parser.add_argument("--image_width",
                          help="Image width.",
                          type=int,
                          default=480)
  app_parser.add_argument("--image_depth",
                          help="Number of color channels.",
                          type=int,
                          default=3)
  app_parser.add_argument("--output_height",
                          help="Output height.",
                          type=int,
                          default=368)
  app_parser.add_argument("--output_width",
                          help="Output width.",
                          type=int,
                          default=480)
  app_parser.add_argument("--resize_side_min",
                          help="The minimal image size in augmentation.",
                          type=int,
                          default=400)
  app_parser.add_argument("--resize_side_max",
                          help="The maximul image size in augmentation.",
                          type=int,
                          default=600)
  app_parser.add_argument("--data_format",
                          help="channels_first or channels_last",
                          default="channels_first")

  # Default configs
  runner_config, callback_config, inputter_config, modeler_config, app_config = \
      config_parser.default_config(parser)

  # Application dependent configs
  callback_config = ImageSegmentationCallbackConfig(
    callback_config,
    num_classes=app_config.num_classes)

  inputter_config = ImageSegmentationInputterConfig(
    inputter_config,
    image_height=app_config.image_height,
    image_width=app_config.image_width,
    image_depth=app_config.image_depth,
    output_height=app_config.output_height,
    output_width=app_config.output_width,
    resize_side_min=app_config.resize_side_min,
    resize_side_max=app_config.resize_side_max,
    num_classes=app_config.num_classes)

  modeler_config = ImageSegmentationModelerConfig(
    modeler_config,
    num_classes=app_config.num_classes,
    data_format=app_config.data_format)

  # Download data if necessary
  downloader.check_and_download(inputter_config)

  if runner_config.mode == "tune":

    inputter_module = importlib.import_module(
      "source.inputter.image_segmentation_csv_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.image_segmentation_modeler")
    runner_module = importlib.import_module(
      "source.runner.parameter_server_runner")

    tuner.tune(app_config,
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
    augmenter = (None if not inputter_config.augmenter else
                 importlib.import_module(
                  "source.augmenter." + inputter_config.augmenter))

    net = importlib.import_module(
      "source.network." + modeler_config.network)

    callbacks = []
    for name in callback_config.callbacks:
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
