"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
"""
import sys
import os
import importlib

"""
Beginner's demo with Resnet32

Train:
python demo/image_classification.py \
--mode=train \
--model_dir=~/demo/model/image_classification_cifar10 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz \
--network=resnet32 \
--augmenter=cifar_augmenter \
--gpu_count=1 --batch_size_per_gpu=128 --epochs=100 \
train_args \
--learning_rate=0.5 --optimizer=momentum \
--piecewise_boundaries=75 \
--piecewise_lr_decay=1.0,0.1 \
--dataset_meta=~/demo/data/cifar10/train.csv

Evaluation:
python demo/image_classification.py \
--mode=eval \
--model_dir=~/demo/model/image_classification_cifar10 \
--network=resnet32 \
--augmenter=cifar_augmenter \
--gpu_count=1 --batch_size_per_gpu=128 --epochs=1 \
eval_args \
--dataset_meta=~/demo/data/cifar10/eval.csv

Infer:
python demo/image_classification.py \
--mode=infer \
--model_dir=~/demo/model/image_classification_cifar10 \
--network=resnet32 \
--augmenter=cifar_augmenter \
--gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
infer_args \
--test_samples=~/demo/data/cifar10/test/appaloosa_s_001975.png,~/demo/data/cifar10/test/domestic_cat_s_001598.png,~/demo/data/cifar10/test/rhea_s_000225.png,~/demo/data/cifar10/test/trucking_rig_s_001216.png

Tune:
python demo/image_classification.py \
--mode=tune \
--model_dir=~/demo/model/image_classification_cifar10 \
--network=resnet32 \
--augmenter=cifar_augmenter \
--gpu_count=1 --batch_size_per_gpu=128 \
tune_args \
--train_dataset_meta=~/demo/data/cifar10/train.csv \
--eval_dataset_meta=~/demo/data/cifar10/eval.csv \
--tune_config=source/tool/ResNet32_CIFAR10_tune_coarse.yaml

python demo/image_classification.py \
--mode=tune \
--model_dir=~/demo/model/image_classification_cifar10 \
--network=resnet32 \
--augmenter=cifar_augmenter \
--gpu_count=1 --batch_size_per_gpu=128 \
tune_args \
--train_dataset_meta=~/demo/data/cifar10/train.csv \
--eval_dataset_meta=~/demo/data/cifar10/eval.csv \
--tune_config=source/tool/ResNet32_CIFAR10_tune_fine.yaml

Pre-trained Model:
curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10-resnet32-20180824.tar.gz | tar xvz -C ~/demo/model

python demo/image_classification.py \
--mode=eval \
--model_dir=~/demo/model/cifar10-resnet32-20180824 \
--network=resnet32 \
--augmenter=cifar_augmenter \
--gpu_count=1 --batch_size_per_gpu=128 --epochs=1 \
eval_args \
--dataset_meta=~/demo/data/cifar10/eval.csv

"""

def main():

  sys.path.append('.')

  from source.tool import downloader
  from source.tool import tuner
  from source.tool import config_parser

  from source.config.image_classification_config import \
      ImageClassificationInputterConfig, \
      ImageClassificationModelerConfig

  parser = config_parser.default_parser()

  parser.add_argument("--num_classes",
                      help="Number of classes.",
                      type=int,
                      default=10)
  parser.add_argument("--image_height",
                      help="Image height.",
                      type=int,
                      default=32)
  parser.add_argument("--image_width",
                      help="Image width.",
                      type=int,
                      default=32)
  parser.add_argument("--image_depth",
                      help="Number of color channels.",
                      type=int,
                      default=3)

  config = parser.parse_args()
  config = config_parser.prepare(config)

  # public_props = (
  #   name for name in dir(config) if not name.startswith('_'))
  # for props_name in public_props:
  #   print(props_name + ": " + str(getattr(config, props_name)))

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

  inputter_config = ImageClassificationInputterConfig(
    inputter_config,
    image_height=config.image_height,
    image_width=config.image_width,
    image_depth=config.image_depth,
    num_classes=config.num_classes)

  modeler_config = ImageClassificationModelerConfig(
    modeler_config,
    num_classes=config.num_classes)

  if config.mode == "tune":

    inputter_module = importlib.import_module(
      "source.inputter.image_classification_csv_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.image_classification_modeler")
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
      "source.inputter.image_classification_csv_inputter").build(
      inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.image_classification_modeler").build(
      modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(
      runner_config, inputter, modeler, callbacks)

    # Run application
    runner.run()


if __name__ == "__main__":
  main()
