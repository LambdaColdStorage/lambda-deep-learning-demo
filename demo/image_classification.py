"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Resnet32

Train:
python demo/image_classification.py \
--num_gpu=4 \
--augmenter_speed_mode \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz

python demo/image_classification.py \
--num_gpu=4 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz

Evaluation:
python demo/image_classification.py --mode=eval \
--num_gpu=4 --epochs=1 \
--augmenter_speed_mode \
--dataset_csv=~/demo/data/cifar10/eval.csv

python demo/image_classification.py --mode=eval \
--num_gpu=4 --epochs=1 \
--dataset_csv=~/demo/data/cifar10/eval.csv

Infer:
python demo/image_classification.py --mode=infer \
--num_gpu=1 --batch_size_per_gpu=1 --epochs=1 \
--test_samples=~/demo/data/cifar10/test/appaloosa_s_001975.png,~/demo/data/cifar10/test/domestic_cat_s_001598.png,~/demo/data/cifar10/test/rhea_s_000225.png,~/demo/data/cifar10/test/trucking_rig_s_001216.png

Tune:
python demo/image_classification.py --mode=tune \
--num_gpu=4

Run with synthetic data:
Train
python demo/image_classification.py \
--inputter=image_classification_syn_inputter \
--augmenter="" \
--num_gpu=4

Evaluation
python demo/image_classification.py --mode=eval \
--inputter=image_classification_syn_inputter \
--augmenter="" \
--num_gpu=4 --epochs=1

Resnet50

python demo/image_classification.py \
--mode=train \
--num_gpu=4 --batch_size_per_gpu=128 --epochs=10 --piecewise_boundaries=10 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--augmenter_speed_mode \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_csv=~/demo/data/StanfordDogs120/train.csv \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
--model_dir=~/demo/model/image_classification_StanfordDog120

python demo/image_classification.py \
--mode=train \
--num_gpu=4 --batch_size_per_gpu=64 --epochs=20000 --piecewise_boundaries=10 \
--network=resnet50 \
--inputter=image_classification_syn_inputter \
--augmenter="" \
--image_height=224 --image_width=224 --num_classes=120 \
--model_dir=~/demo/model/image_classification_StanfordDog120

Transfer Learning:

wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz  \
-O /tmp/resnet_v2_50_2017_04_14.tar.gz && if ! [ -d "~/demo/model/resnet_v2_50_2017_04_14" ]; then mkdir ~/demo/model/resnet_v2_50_2017_04_14; \
fi && tar -xzf /tmp/resnet_v2_50_2017_04_14.tar.gz --directory=${HOME}/demo/model/resnet_v2_50_2017_04_14 && rm -rf /tmp/resnet_v2_50_2017_04_14.tar.gz

Train with real data:
python demo/image_classification.py \
--mode=train \
--num_gpu=4 --batch_size_per_gpu=64 --epochs=20 --piecewise_boundaries=10 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--augmenter_speed_mode=True \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_csv=~/demo/data/StanfordDogs120/train.csv \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
--model_dir=~/demo/model/image_classification_StanfordDog120 \
--pretrained_dir=~/demo/model/resnet_v2_50_2017_04_14 \
--skip_pretrained_var_list="resnet_v2_50/logits,global_step" \
--trainable_var_list="resnet_v2_50/logits"

Train with synthetic data:

python demo/image_classification.py \
--mode=train \
--num_gpu=4 --batch_size_per_gpu=256 --epochs=20000 --piecewise_boundaries=10 \
--network=resnet50 \
--inputter=image_classification_syn_inputter \
--augmenter="" \
--image_height=224 --image_width=224 --num_classes=120 \
--model_dir=~/demo/model/image_classification_StanfordDog120 \
--pretrained_dir=~/demo/model/resnet_v2_50_2017_04_14 \
--skip_pretrained_var_list="resnet_v2_50/logits,global_step" \
--trainable_var_list="resnet_v2_50/logits"

python demo/image_classification.py \
--mode=train \
--num_gpu=4 --batch_size_per_gpu=256 --epochs=20000 --piecewise_boundaries=10 \
--network=resnet50 \
--inputter=image_classification_syn_inputter \
--augmenter=vgg_augmenter \
--augmenter_speed_mode=False \
--image_height=224 --image_width=224 --num_classes=120 \
--model_dir=~/demo/model/image_classification_StanfordDog120 \
--pretrained_dir=~/demo/model/resnet_v2_50_2017_04_14 \
--skip_pretrained_var_list="resnet_v2_50/logits,global_step" \
--trainable_var_list="resnet_v2_50/logits"

python demo/image_classification.py \
--mode=eval \
--num_gpu=4 --epochs=1 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_csv=~/demo/data/StanfordDogs120/eval.csv \
--model_dir=~/demo/model/image_classification_StanfordDog120
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
  parser.add_argument("--augmenter_speed_mode",
                      action='store_true',
                      help="Flag to use speed mode in augmentation")
  parser.add_argument("--network", choices=["resnet32", "resnet50"],
                      type=str,
                      help="Choose a network architecture",
                      default="resnet32")
  parser.add_argument("--mode", choices=["train", "eval", "infer", "tune"],
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
                      default="channels_last")
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
  parser.add_argument("--dataset_url",
                      help="URL for downloading data",
                      default="")
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
                      default="")
  parser.add_argument("--skip_l2_loss_vars",
                      help="List of blacklisted trainable Variables for L2 regularization.",
                      type=str,
                      default="BatchNorm,preact,postnorm")

  args = parser.parse_args()

  args = args_parser.prepare(args)

  # Download data if necessary
  if args.inputter == "image_classification_syn_inputter":
    print("Use synthetic data")
  else:
    if args.mode != "infer":
      if not os.path.exists(args.dataset_csv):
        downloader.download_and_extract(args.dataset_csv,
                                        args.dataset_url, False)
      else:
        print("Found " + args.dataset_csv + ".")

  if args.mode == "tune":
    tuner.tune(args)
  else:
    demo = app.APP(args)
    demo.run()


if __name__ == "__main__":
  main()
