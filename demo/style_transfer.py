"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
"""
import sys
import os
import importlib

"""
Style Transfer Demo
"""


def main():

  sys.path.append('.')

  from source.tool import downloader
  from source.tool import tuner
  from source.tool import config_parser

  from source.config.style_transfer_config import \
      StyleTransferInputterConfig, StyleTransferModelerConfig

  parser = config_parser.default_parser()

  app_parser = parser.add_argument_group('app')

  app_parser.add_argument("--style_weight",
                          help="Weight for style loss",
                          default=100)
  app_parser.add_argument("--content_weight",
                          help="Weight for content loss",
                          default=7.5)
  app_parser.add_argument("--tv_weight",
                          help="Weight for tv loss",
                          default=200)
  app_parser.add_argument("--image_height",
                          help="Image height.",
                          type=int,
                          default=256)
  app_parser.add_argument("--image_width",
                          help="Image width.",
                          type=int,
                          default=256)
  app_parser.add_argument("--resize_side_min",
                          help="The minimal image size in augmentation.",
                          type=int,
                          default=400)
  app_parser.add_argument("--resize_side_max",
                          help="The maximul image size in augmentation.",
                          type=int,
                          default=600)
  app_parser.add_argument("--image_depth",
                          help="Number of color channels.",
                          type=int,
                          default=3)
  app_parser.add_argument("--feature_net",
                          help="Name of feature net",
                          default="vgg_19_conv")
  app_parser.add_argument("--feature_net_path",
                          help="Path to pre-trained vgg model.",
                          default=os.path.join(
                          os.environ['HOME'],
                          "demo/model/vgg_19_2016_08_28/vgg_19.ckpt"))
  app_parser.add_argument("--style_image_path",
                          help="Path to style image",
                          default=os.path.join(
                          os.environ['HOME'], "demo/data/mscoco_fns/gothic.jpg"))
  app_parser.add_argument("--data_format",
                          help="channels_first or channels_last",
                          choices=["channels_first", "channels_last"],
                          default="channels_last")
  app_parser.add_argument("--feature_net_url",
                          help="URL for downloading pre-trained feature_net",
                          default="http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz")

  # Default configs
  runner_config, callback_config, inputter_config, modeler_config, app_config = \
      config_parser.default_config(parser)

  # Application dependent configs
  inputter_config = StyleTransferInputterConfig(
    inputter_config,
    image_height=app_config.image_height,
    image_width=app_config.image_width,
    image_depth=app_config.image_depth,
    resize_side_min=app_config.resize_side_min,
    resize_side_max=app_config.resize_side_max)

  modeler_config = StyleTransferModelerConfig(
    modeler_config,
    data_format=app_config.data_format,
    image_depth=app_config.image_depth,
    style_weight=app_config.style_weight,
    content_weight=app_config.content_weight,
    tv_weight=app_config.tv_weight,
    feature_net=app_config.feature_net,
    feature_net_path=app_config.feature_net_path,
    style_image_path=app_config.style_image_path)

  # Download data if necessary
  downloader.check_and_download(inputter_config)

  if runner_config.mode == "tune":

    inputter_module = importlib.import_module(
      "source.inputter.style_transfer_csv_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.style_transfer_modeler")
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
      "source.inputter.style_transfer_csv_inputter").build(
      inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.style_transfer_modeler").build(
      modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(
      runner_config, inputter, modeler, callbacks)

    # Run application
    runner.run()


if __name__ == "__main__":
  main()
