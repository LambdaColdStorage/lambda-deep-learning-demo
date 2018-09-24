"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
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

  from source.config.text_generation_config import \
    TextGenerationCallbackConfig, TextGenerationInputterConfig, \
    TextGenerationModelerConfig
  parser = config_parser.default_parser()


  # parser.add_argument("--network", choices=["char_rnn"],
  #                     type=str,
  #                     help="Choose a network architecture",
  #                     default="char_rnn")
  # parser.add_argument("--dataset_url",
  #                     help="URL for downloading data",
  #                     default="https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz")
  # parser.add_argument("--train_callbacks",
  #                     help="List of callbacks in training.",
  #                     type=str,
  #                     default="train_basic,train_loss,train_accuracy,train_speed,train_summary")
  # parser.add_argument("--eval_callbacks",
  #                     help="List of callbacks in evaluation.",
  #                     type=str,
  #                     default="eval_basic,eval_loss,eval_accuracy,eval_speed,eval_summary")
  # parser.add_argument("--infer_callbacks",
  #                     help="List of callbacks in inference.",
  #                     type=str,
  #                     default="infer_basic,infer_display_text_generation")

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

  if config.mode == "tune":

    inputter_module = importlib.import_module(
      "source.inputter.text_generation_txt_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.text_generation_modeler")
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
      "source.inputter.text_generation_txt_inputter").build(
      inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.text_generation_modeler").build(
      modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(
      runner_config, inputter, modeler, callbacks)

    # Run application
    runner.run()


if __name__ == "__main__":
  main()
