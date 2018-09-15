"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
Train:
python demo/image_segmentation.py --mode=train \
--gpu_count=4 --batch_size_per_gpu=16 --epochs=200 \
--piecewise_boundaries=100 \
--piecewise_lr_decay=1.0,0.1 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz \
--dataset_meta=~/demo/data/camvid/train.csv \
--model_dir=~/demo/model/image_segmentation_camvid

Evaluation:
python demo/image_segmentation.py --mode=eval \
--gpu_count=4 --batch_size_per_gpu=16 --epochs=1 \
--dataset_meta=~/demo/data/camvid/val.csv \
--model_dir=~/demo/model/image_segmentation_camvid

Infer:
python demo/image_segmentation.py --mode=infer \
--batch_size_per_gpu=1 --epochs=1 --gpu_count=1 \
--model_dir=~/demo/model/image_segmentation_camvid \
--test_samples=~/demo/data/camvid/test/0001TP_008550.png,~/demo/data/camvid/test/Seq05VD_f02760.png,~/demo/data/camvid/test/Seq05VD_f04650.png,~/demo/data/camvid/test/Seq05VD_f05100.png

Tune:
python demo/image_segmentation.py --mode=tune \
--dataset_meta=~/demo/data/camvid/train.csv \
--model_dir=~/demo/model/image_segmentation_camvid \
--gpu_count=4
"""
import sys
import os
import argparse
import importlib


def main():

  sys.path.append('.')

  from source import app
  from source.tool import downloader
  from source.tool import tuner
  from source.tool import config_parser

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--inputter",
                      type=str,
                      help="Name of the inputter",
                      default="image_segmentation_csv_inputter")
  parser.add_argument("--modeler",
                      type=str,
                      help="Name of the modeler",
                      default="image_segmentation_modeler")
  parser.add_argument("--runner",
                      type=str,
                      help="Name of the runner",
                      default="parameter_server_runner")
  parser.add_argument("--augmenter",
                      type=str,
                      help="Name of the augmenter",
                      default="fcn_augmenter")
  parser.add_argument("--augmenter_speed_mode",
                      action='store_true',
                      help="Flag to use speed mode in augmentation")
  parser.add_argument("--network", choices=["fcn"],
                      type=str,
                      help="Choose a network architecture",
                      default="fcn")
  parser.add_argument("--mode", choices=["train", "eval", "infer", "tune"],
                      type=str,
                      help="Choose a job mode from train, eval, and infer.",
                      default="train")
  parser.add_argument("--dataset_meta", type=str,
                      help="Path to dataset's csv meta file",
                      default="")
  parser.add_argument("--batch_size_per_gpu",
                      help="Number of images on each GPU.",
                      type=int,
                      default=16)
  parser.add_argument("--gpu_count",
                      help="Number of GPUs.",
                      type=int,
                      default=4)
  parser.add_argument("--shuffle_buffer_size",
                      help="Buffer size for shuffling training images.",
                      type=int,
                      default=1000)
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
  parser.add_argument("--image_depth",
                      help="Number of color channels.",
                      type=int,
                      default=3)
  parser.add_argument("--data_format",
                      help="channels_first or channels_last",
                      default="channels_first")
  parser.add_argument("--model_dir",
                      help="Directory to save mode",
                      type=str,
                      default=os.path.join(os.environ['HOME'],
                                           "demo/model/image_segmentation_camvid"))
  parser.add_argument("--l2_weight_decay",
                      help="Weight decay for L2 regularization in training",
                      type=float,
                      default=0.0002)
  parser.add_argument("--learning_rate",
                      help="Initial learning rate in training.",
                      type=float,
                      default=0.1)
  parser.add_argument("--epochs",
                      help="Number of epochs.",
                      type=int,
                      default=200)
  parser.add_argument("--piecewise_boundaries",
                      help="Epochs to decay learning rate",
                      default="100")
  parser.add_argument("--piecewise_lr_decay",
                      help="Decay ratio for learning rate",
                      default="1.0,0.1")
  parser.add_argument("--optimizer",
                      help="Name of optimizer",
                      choices=["adadelta", "adagrad", "adam", "ftrl",
                               "momentum", "rmsprop", "sgd"],
                      default="momentum")
  parser.add_argument("--log_every_n_iter",
                      help="Number of steps to log",
                      type=int,
                      default=2)
  parser.add_argument("--save_summary_steps",
                      help="Number of steps to save summary.",
                      type=int,
                      default=2)
  parser.add_argument("--save_checkpoints_steps",
                      help="Number of steps to save checkpoints",
                      type=int,
                      default=100)
  parser.add_argument("--keep_checkpoint_max",
                      help="Maximum number of checkpoints to save.",
                      type=int,
                      default=1)
  parser.add_argument("--class_names",
                      help="List of class names.",
                      default="")
  parser.add_argument("--test_samples",
                      help="A string of comma seperated testing data. "
                      "Must be provided for infer mode.",
                      type=str)
  parser.add_argument("--summary_names",
                      help="A string of comma seperated names for summary",
                      type=str,
                      default="loss,accuracy,learning_rate")
  parser.add_argument("--dataset_url",
                      help="URL for downloading data",
                      default="https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz")
  parser.add_argument("--pretrained_dir",
                      help="Path to pretrained network (for transfer learning).",
                      type=str,
                      default="")
  parser.add_argument("--skip_pretrained_var",
                      help="Variables to skip in restoring from pretrained model (for transfer learning).",
                      type=str,
                      default="")
  parser.add_argument("--trainable_vars",
                      help="List of trainable Variables. \
                           If None, all variables in tf.GraphKeys.TRAINABLE_VARIABLES \
                           will be trained, apart from the ones blacklisted by skip_trainable_vars.",
                      type=str,
                      default="")
  parser.add_argument("--skip_trainable_vars",
                      help="List of blacklisted trainable Variables.",
                      type=str,
                      default="")
  parser.add_argument("--skip_l2_loss_vars",
                      help="List of blacklisted trainable Variables \
                            for L2 regularization.",
                      type=str,
                      default="BatchNorm,preact,postnorm")
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

  if config.mode == "tune":
    tuner.tune(config)
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
      callback_names = config.train_callbacks.split(",")
    elif config.mode == "eval":
      callback_names = config.eval_callbacks.split(",")
    elif config.mode == "infer":
      callback_names = config.infer_callbacks.split(",")

    callbacks = []
    for name in callback_names:
      callback = importlib.import_module(
        "source.callback." + name).build(config)
      callbacks.append(callback)

    inputter = importlib.import_module(
      "source.inputter." + config.inputter).build(config, augmenter)

    modeler = importlib.import_module(
      "source.modeler." + config.modeler).build(config, net)

    runner = importlib.import_module(
      "source.runner." + config.runner).build(config, inputter, modeler, callbacks)

    demo = app.APP(runner)
    demo.run()


if __name__ == "__main__":
  main()
