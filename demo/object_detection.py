"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

python demo/object_detection.py \
--mode=train \
--model_dir=~/demo/model/rpn_mscoco \
--network=rpn \
--batch_size_per_gpu=4 --epochs=1 \
--dataset_dir=/media/chuan/cb8101e3-b1d2-4f5c-a4cd-7badb0dd6800/data/mscoco \
train_args \
--learning_rate=0.5 --optimizer=momentum \
--piecewise_boundaries=50,75,90 \
--piecewise_lr_decay=1.0,0.1,0.01,0.001 \
--dataset_meta=valminusminival2014
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

  from source.config.object_detection_config import \
      ObjectDetectionInputterConfig, \
      ObjectDetectionModelerConfig

  parser = config_parser.default_parser()

  parser.add_argument("--num_classes",
                      help="Number of classes.",
                      type=int,
                      default=81)
  parser.add_argument("--dataset_dir",
                      help="Path to dataset.",
                      type=str,
                      default="/media/chuan/cb8101e3-b1d2-4f5c-a4cd-7badb0dd6800/data/mscoco")

  config = parser.parse_args()

  config = config_parser.prepare(config)

  # Generate config
  runner_config, callback_config, inputter_config, modeler_config = \
      config_parser.default_config(config)

  inputter_config = ObjectDetectionInputterConfig(
    inputter_config,
    dataset_dir=config.dataset_dir)

  modeler_config = ObjectDetectionModelerConfig(
    modeler_config)

  if config.mode == "tune":
    pass
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
      "source.inputter.object_detection_mscoco_inputter").build(
      inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.object_detection_modeler").build(
      modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(
      runner_config, inputter, modeler, callbacks)

    # Run application
    runner.dev()

if __name__ == "__main__":
  main()