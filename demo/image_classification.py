"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
Train:
python demo/image_classification.py \
--num_gpu=4

Evaluation:
python demo/image_classification.py --mode=eval \
--num_gpu=4 --epochs=1 \
--dataset_csv=~/demo/data/cifar10/eval.csv

Infer:
python demo/image_classification.py --mode=infer \
--num_gpu=1 --batch_size_per_gpu=1 --epochs=1 \
--test_samples=~/demo/data/cifar10/test/appaloosa_s_001975.png,~/demo/data/cifar10/test/domestic_cat_s_001598.png,~/demo/data/cifar10/test/rhea_s_000225.png,~/demo/data/cifar10/test/trucking_rig_s_001216.png
"""
import os
import argparse

import app


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--inputter",
                      type=str,
                      help="Name of the inputter",
                      default="image_classification_csv_inputter")
  parser.add_argument("--modeler",
                      type=str,
                      help="Name of the modeler",
                      default="image_classification_modeler")
  parser.add_argument("--runner",
                      type=str,
                      help="Name of the runner",
                      default="parameter_server_runner")
  parser.add_argument("--augmenter",
                      type=str,
                      help="Name of the augmenter",
                      default="cifar_augmenter")
  parser.add_argument("--network", choices=["resnet32"],
                      type=str,
                      help="Choose a network architecture",
                      default="resnet32")
  parser.add_argument("--mode", choices=["train", "eval", "infer"],
                      type=str,
                      help="Choose a job mode from train, eval, and infer.",
                      default="train")
  parser.add_argument("--dataset_csv", type=str,
                      help="Path to dataset's csv meta file",
                      default=os.path.join(os.environ['HOME'],
                                           "demo/data/cifar10/train.csv"))
  parser.add_argument("--batch_size_per_gpu",
                      help="Number of images on each GPU.",
                      type=int,
                      default=128)
  parser.add_argument("--num_gpu",
                      help="Number of GPUs.",
                      type=int,
                      default=4)
  parser.add_argument("--epochs",
                      help="Number of epochs.",
                      type=int,
                      default=5)
  parser.add_argument("--shuffle_buffer_size",
                      help="Buffer size for shuffling training images.",
                      type=int,
                      default=2000)
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
  parser.add_argument("--data_format",
                      help="channels_first or channels_last",
                      default="channels_first")
  parser.add_argument("--model_dir",
                      help="Directory to save mode",
                      type=str,
                      default=os.path.join(os.environ['HOME'],
                                           "demo/model/image_classification_cifar10"))
  parser.add_argument("--l2_weight_decay",
                      help="Weight decay for L2 regularization in training",
                      type=float,
                      default=0.0002)
  parser.add_argument("--learning_rate",
                      help="Initial learning rate in training.",
                      type=float,
                      default=0.5)
  parser.add_argument("--piecewise_boundaries",
                      help="Epochs to decay learning rate",
                      default="2")
  parser.add_argument("--piecewise_learning_rate_decay",
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
                      default="airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck")
  parser.add_argument("--test_samples",
                      help="A string of comma seperated testing data. "
                      "Must be provided for infer mode.",
                      type=str)
  parser.add_argument("--summary_names",
                      help="A string of comma seperated names for summary",
                      type=str,
                      default="loss,accuracy,learning_rate")
  args = parser.parse_args()

  args.dataset_csv = os.path.expanduser(args.dataset_csv)
  args.model_dir = os.path.expanduser(args.model_dir)
  args.summary_names = args.summary_names.split(",")

  demo = app.APP(args)

  demo.run()


if __name__ == "__main__":
  main()
