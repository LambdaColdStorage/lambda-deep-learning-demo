"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
"""
import sys
import os
import importlib

"""
Object Detection Demo
"""


def main():

  sys.path.append('.')

  from source.tool import tuner
  from source.tool import config_parser

  from source.config.object_detection_config import \
      ObjectDetectionInputterConfig, \
      ObjectDetectionModelerConfig

  parser = config_parser.default_parser()

  app_parser = parser.add_argument_group('app')

  app_parser.add_argument("--num_classes",
                          help="Number of classes.",
                          type=int,
                          default=81)
  app_parser.add_argument("--resolution",
                          help="Image resolution used for detectoin.",
                          type=int,
                          default=512) 
  app_parser.add_argument("--dataset_dir",
                          help="Path to dataset.",
                          type=str,
                          default="/mnt/data/data/mscoco")
  app_parser.add_argument("--feature_net",
                          help="Name of feature net",
                          default="vgg_16_reduced")
  app_parser.add_argument("--feature_net_path",
                          help="Path to pre-trained vgg model.",
                          default=os.path.join(
                          os.environ['HOME'],
                          "demo/model/vgg_16_2016_08_28/vgg_16.ckpt"))
  app_parser.add_argument("--data_format",
                          help="channels_first or channels_last",
                          choices=["channels_first", "channels_last"],
                          default="channels_last")

  # Default configs
  runner_config, callback_config, inputter_config, modeler_config, app_config = \
      config_parser.default_config(parser)

  inputter_config = ObjectDetectionInputterConfig(
    inputter_config,
    dataset_dir=app_config.dataset_dir,
    num_classes=app_config.num_classes,
    resolution=app_config.resolution)

  modeler_config = ObjectDetectionModelerConfig(
    modeler_config,
    num_classes=app_config.num_classes,
    data_format=app_config.data_format,
    feature_net=app_config.feature_net,
    feature_net_path=app_config.feature_net_path)

  if runner_config.mode == "tune":
    inputter_module = importlib.import_module(
      "source.inputter.object_detection_mscoco_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.object_detection_modeler")
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
      "source.inputter.object_detection_mscoco_inputter").build(
      inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.object_detection_modeler").build(
      modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(
      runner_config, inputter, modeler, callbacks)

    # Run application
    runner.run()


if __name__ == "__main__":
  main()
