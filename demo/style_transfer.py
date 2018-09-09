"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
Train:
python demo/style_transfer.py \
--num_gpu=1

Eval:
python demo/style_transfer.py --mode=eval \
--num_gpu=1 --epochs=1 \
--dataset_csv=~/demo/data/mscoco_fns/eval2014.csv

Infer:
python demo/style_transfer.py --mode=infer \
--batch_size_per_gpu=1 --epochs=1 --num_gpu=1 \
--test_samples=~/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg

Tune:
python demo/style_transfer.py --mode=tune \
--num_gpu=1
"""
import os
import argparse

import app
from tool import downloader
from tool import tuner
from tool import args_parser


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--inputter",
                      type=str,
                      help="Name of the inputter",
                      default="style_transfer_csv_inputter")
  parser.add_argument("--modeler",
                      type=str,
                      help="Name of the modeler",
                      default="style_transfer_modeler")
  parser.add_argument("--runner",
                      type=str,
                      help="Name of the runner",
                      default="parameter_server_runner")
  parser.add_argument("--augmenter",
                      type=str,
                      help="Name of the augmenter",
                      default="fns_augmenter")
  parser.add_argument("--network", choices=["fns"],
                      type=str,
                      help="Choose a network architecture",
                      default="fns")
  parser.add_argument("--mode", choices=["train", "eval", "infer", "tune"],
                      type=str,
                      help="Choose a job mode from train, eval, and infer.",
                      default="train")
  parser.add_argument("--style_weight",
                      help="Weight for style loss",
                      default=100)
  parser.add_argument("--content_weight",
                      help="Weight for content loss",
                      default=15)
  parser.add_argument("--tv_weight",
                      help="Weight for tv loss",
                      default=200)
  parser.add_argument("--dataset_csv", type=str,
                      help="Path to dataset's csv meta file",
                      default=os.path.join(os.environ['HOME'],
                                           "demo/data/mscoco_fns/train2014.csv"))
  parser.add_argument("--batch_size_per_gpu",
                      help="Number of images on each GPU.",
                      type=int,
                      default=4)
  parser.add_argument("--num_gpu",
                      help="Number of GPUs.",
                      type=int,
                      default=4)
  parser.add_argument("--shuffle_buffer_size",
                      help="Buffer size for shuffling training images.",
                      type=int,
                      default=1000)
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
                      default=os.path.join(os.environ['HOME'],
                                           "demo/data/mscoco_fns/gothic.jpg"))

  parser.add_argument("--data_format",
                      help="channels_first or channels_last",
                      choices=["channels_first", "channels_last"],
                      default="channels_last")
  parser.add_argument("--model_dir",
                      help="Directory to save mode",
                      type=str,
                      default=os.path.join(os.environ['HOME'],
                                           "demo/model/style_transfer_mscoco_fns"))
  parser.add_argument("--l2_weight_decay",
                      help="Weight decay for L2 regularization in training",
                      type=float,
                      default=0.0002)
  parser.add_argument("--learning_rate",
                      help="Initial learning rate in training.",
                      type=float,
                      default=0.01)
  parser.add_argument("--epochs",
                      help="Number of epochs.",
                      type=int,
                      default=10)
  parser.add_argument("--piecewise_boundaries",
                      help="Epochs to decay learning rate",
                      default="5")
  parser.add_argument("--piecewise_learning_rate_decay",
                      help="Decay ratio for learning rate",
                      default="1.0,0.1")
  parser.add_argument("--optimizer",
                      help="Name of optimizer",
                      choices=["adadelta", "adagrad", "adam", "ftrl",
                               "momentum", "rmsprop", "sgd"],
                      default="rmsprop")
  parser.add_argument("--log_every_n_iter",
                      help="Number of steps to log",
                      type=int,
                      default=2)
  parser.add_argument("--save_summary_steps",
                      help="Number of steps to save summary.",
                      type=int,
                      default=10)
  parser.add_argument("--save_checkpoints_steps",
                      help="Number of steps to save checkpoints",
                      type=int,
                      default=100)
  parser.add_argument("--keep_checkpoint_max",
                      help="Maximum number of checkpoints to save.",
                      type=int,
                      default=1)
  parser.add_argument("--test_samples",
                      help="A string of comma seperated testing data. "
                      "Must be provided for infer mode.",
                      type=str)
  parser.add_argument("--summary_names",
                      help="A string of comma seperated names for summary",
                      type=str,
                      default="loss,learning_rate")
  parser.add_argument("--dataset_url",
                      help="URL for downloading data",
                      default="https://s3-us-west-2.amazonaws.com/lambdalabs-files/mscoco_fns.tar.gz")
  parser.add_argument("--feature_net_url",
                      help="URL for downloading pre-trained feature_net",
                      default="http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz")
  parser.add_argument("--pretrained_dir",
                      help="Path to pretrained network (for transfer learning).",
                      type=str,
                      default="")
  parser.add_argument("--skip_pretrained_var_list",
                      help="Variables to skip in restoring from pretrained model (for transfer learning).",
                      type=str,
                      default="")
  parser.add_argument("--trainable_var_list",
                      help="List of trainable Variables. \
                           If None all variables in tf.GraphKeys.TRAINABLE_VARIABLES \
                           will be trained, subjected to the ones blacklisted by skip_trainable_var_list.",
                      type=str,
                      default="")
  parser.add_argument("--skip_trainable_var_list",
                      help="List of blacklisted trainable Variables.",
                      type=str,
                      default="vgg_19")
  parser.add_argument("--skip_l2_loss_vars",
                      help="List of blacklisted trainable Variables for L2 regularization.",
                      type=str,
                      default="")

  args = parser.parse_args()

  args = args_parser.prepare(args)

  # Download data if necessary
  if not os.path.exists(args.dataset_csv):
    downloader.download_and_extract(args.dataset_csv,
                                    args.dataset_url, False)
  else:
    print("Found " + args.dataset_csv + ".")

  if not os.path.exists(args.feature_net_path):
    downloader.download_and_extract(args.feature_net_path,
                                    args.feature_net_url, True)
  else:
    print("Found " + args.feature_net_path + '.')

  if args.mode == "tune":
    tuner.tune(args)
  else:
    demo = app.APP(args)
    demo.run()


if __name__ == "__main__":
  main()
