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

  from source.config.style_transfer_config import \
      StyleTransferInputterConfig, StyleTransferModelerConfig

  parser = config_parser.default_parser()
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
                      default=os.path.join(
                        os.environ['HOME'], "demo/data/mscoco_fns/gothic.jpg"))
  parser.add_argument("--data_format",
                      help="channels_first or channels_last",
                      choices=["channels_first", "channels_last"],
                      default="channels_last")
  parser.add_argument("--feature_net_url",
                      help="URL for downloading pre-trained feature_net",
                      default="http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz")

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

  inputter_config = StyleTransferInputterConfig(
    inputter_config,
    image_height=config.image_height,
    image_width=config.image_width,
    image_depth=config.image_depth,
    resize_side_min=config.resize_side_min,
    resize_side_max=config.resize_side_max)

  modeler_config = StyleTransferModelerConfig(
    modeler_config,
    data_format=config.data_format,
    image_depth=config.image_depth,
    style_weight=config.style_weight,
    content_weight=config.content_weight,
    tv_weight=config.tv_weight,
    feature_net=config.feature_net,
    feature_net_path=config.feature_net_path,
    style_image_path=config.style_image_path)

  if config.mode == "tune":

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

    callbacks = []
    for name in config.callbacks:
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
