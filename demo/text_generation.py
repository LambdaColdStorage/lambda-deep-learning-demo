"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Train:
python demo/text_generation.py --mode=train \
--gpu_count=1 --batch_size_per_gpu=128 --epochs=20 \
--learning_rate=0.002 --optimizer=adam \
--piecewise_boundaries=10 \
--piecewise_lr_decay=1.0,0.1 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
--dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
--model_dir=~/demo/model/text_gen_shakespeare \
--summary_names=loss,learning_rate

Infer:
python demo/text_generation.py --mode=infer \
--gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
--dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
--model_dir=~/demo/model/text_gen_shakespeare

Tune:
python demo/text_generation.py --mode=tune \
--gpu_count=1 \
--dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
--model_dir=~/demo/model/text_gen_shakespeare
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

  parser.add_argument("--augmenter",
                      type=str,
                      help="Name of the augmenter",
                      default=None)
  parser.add_argument("--network", choices=["char_rnn"],
                      type=str,
                      help="Choose a network architecture",
                      default="char_rnn")
  parser.add_argument("--dataset_url",
                      help="URL for downloading data",
                      default="https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz")
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
                      default="infer_basic,infer_display_text_generation")

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

  if config.mode == "tune":

    augmenter = (None if not config.augmenter else
                 importlib.import_module(
                  "source.augmenter." + config.augmenter))

    net = getattr(importlib.import_module(
      "source.network." + config.network), "net")

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
