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
python demo/image_classification.py --mode=train \
--gpu_count=4 --batch_size_per_gpu=384 --epochs=100 \
--learning_rate=0.5 --optimizer=momentum \
--piecewise_boundaries=75 \
--piecewise_lr_decay=1.0,0.1 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz \
--dataset_meta=~/demo/data/cifar10/train.csv \
--model_dir=~/demo/model/image_classification_cifar10

Evaluation:
python demo/image_classification.py --mode=eval \
--gpu_count=4 --batch_size_per_gpu=256 --epochs=1 \
--dataset_meta=~/demo/data/cifar10/eval.csv \
--model_dir=~/demo/model/image_classification_cifar10

Infer:
python demo/image_classification.py --mode=infer \
--gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
--model_dir=~/demo/model/image_classification_cifar10 \
--test_samples=~/demo/data/cifar10/test/appaloosa_s_001975.png,~/demo/data/cifar10/test/domestic_cat_s_001598.png,~/demo/data/cifar10/test/rhea_s_000225.png,~/demo/data/cifar10/test/trucking_rig_s_001216.png

Tune:
python demo/image_classification.py --mode=tune \
--dataset_meta=~/demo/data/cifar10/train.csv \
--model_dir=~/demo/model/image_classification_cifar10 \
--gpu_count=4 \
--tune_config=source/tool/ResNet32_CIFAR10_tune_coarse.yaml

python demo/image_classification.py --mode=tune \
--dataset_meta=~/demo/data/cifar10/train.csv \
--model_dir=~/demo/model/image_classification_cifar10 \
--gpu_count=4 \
--tune_config=source/tool/ResNet32_CIFAR10_tune_fine.yaml

Pre-trained Model:
curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10-resnet32-20180824.tar.gz | tar xvz -C ~/demo/model

python demo/image_classification.py --mode=eval \
--gpu_count=4 --batch_size_per_gpu=256 --epochs=1 \
--dataset_meta=~/demo/data/cifar10/eval.csv \
--model_dir=~/demo/model/cifar10-resnet32-20180824
"""

"""
Transfer Learning with ResNet50

Prepare data:
(mkdir ~/demo/model/resnet_v2_50_2017_04_14;
curl http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz | tar xvz -C ~/demo/model/resnet_v2_50_2017_04_14)

Train:
python demo/image_classification.py --mode=train \
--gpu_count=4 --batch_size_per_gpu=16 --epochs=50 \
--optimizer=adam \
--learning_rate=0.0025 \
--piecewise_boundaries=25 \
--piecewise_lr_decay=1.0,0.1 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/train.csv \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
--model_dir=~/demo/model/image_classification_StanfordDogs120 \
--pretrained_model=~/demo/model/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt \
--skip_pretrained_var=resnet_v2_50/logits,global_step,power \
--trainable_vars=resnet_v2_50/logits

Evaluation:
python demo/image_classification.py \
--mode=eval \
--gpu_count=1 --batch_size_per_gpu=16 --epochs=1 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/eval.csv \
--model_dir=~/demo/model/image_classification_StanfordDogs120
"""

"""
Transfer Learning with Inception V4

Prepare data:
(mkdir ~/demo/model/inception_v4_2016_09_09;
curl http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz | tar xvz -C ~/demo/model/inception_v4_2016_09_09)

python demo/image_classification.py --mode=train \
--gpu_count=4 --batch_size_per_gpu=16 --epochs=10 \
--optimizer=adam \
--learning_rate=0.0025 \
--piecewise_boundaries=5 \
--piecewise_lr_decay=1.0,0.1 \
--network=inception_v4 \
--augmenter=inception_augmenter \
--image_height=299 --image_width=299 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/train.csv \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
--model_dir=~/demo/model/image_classification_StanfordDogs120 \
--pretrained_model=~/demo/model/inception_v4_2016_09_09/inception_v4.ckpt \
--skip_pretrained_var=InceptionV4/AuxLogits,InceptionV4/Logits,global_step,power \
--trainable_vars=InceptionV4/AuxLogits,InceptionV4/Logits

Evaluation:
python demo/image_classification.py \
--mode=eval \
--gpu_count=1 --batch_size_per_gpu=16 --epochs=1 \
--network=inception_v4 \
--augmenter=inception_augmenter \
--image_height=299 --image_width=299 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/eval.csv \
--model_dir=~/demo/model/image_classification_StanfordDogs120
"""

"""
Transfer Learning with NasNet-A-Large

Prepare data:
(mkdir ~/demo/model/nasnet-a_large_04_10_2017;
curl https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz | tar xvz -C ~/demo/model/nasnet-a_large_04_10_2017)

python demo/image_classification.py --mode=train \
--gpu_count=4 --batch_size_per_gpu=16 --epochs=10 \
--optimizer=adam \
--learning_rate=0.0025 \
--piecewise_boundaries=5 \
--piecewise_lr_decay=1.0,0.1 \
--network=nasnet_A_large \
--augmenter=inception_augmenter \
--image_height=331 --image_width=331 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/train.csv \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
--model_dir=~/demo/model/image_classification_StanfordDogs120 \
--pretrained_model=~/demo/model/nasnet-a_large_04_10_2017/model.ckpt \
--skip_pretrained_var=final_layer,aux_logits,global_step,power \
--trainable_vars=final_layer,aux_logits \
--skip_l2_loss_vars=gamma,beta

Evaluation:
python demo/image_classification.py \
--mode=eval \
--gpu_count=1 --batch_size_per_gpu=16 --epochs=1 \
--network=nasnet_A_large \
--augmenter=inception_augmenter \
--image_height=331 --image_width=331 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/eval.csv \
--model_dir=~/demo/model/image_classification_StanfordDogs120

Tune:
python demo/image_classification.py --mode=tune \
--network=nasnet_A_large \
--augmenter=inception_augmenter \
--image_height=331 --image_width=331 --num_classes=120 \
--gpu_count=4 --batch_size_per_gpu=16 \
--dataset_meta=~/demo/data/StanfordDogs120/train.csv \
--model_dir=~/demo/model/image_classification_StanfordDogs120 \
--pretrained_model=~/demo/model/nasnet-a_large_04_10_2017/model.ckpt \
--skip_pretrained_var=final_layer,aux_logits,global_step,power \
--trainable_vars=final_layer,aux_logits \
--skip_l2_loss_vars=gamma,beta \
--tune_config=source/tool/NasNet_A_Large_StanfordDogs120_tune.yaml
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

  parser.add_argument("--augmenter",
                      type=str,
                      help="Name of the augmenter",
                      default="cifar_augmenter")
  parser.add_argument("--network", choices=["resnet32", "resnet50", "inception_v4", "nasnet_A_large"],
                      type=str,
                      help="Choose a network architecture",
                      default="resnet32")
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
  parser.add_argument("--dataset_url",
                      help="URL for downloading data",
                      default="")
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
                      default="infer_basic,infer_display_image_classification")

  config = parser.parse_args()

  config = config_parser.prepare(config)

  # Download data if necessary
  if config.mode != "infer":
    if not os.path.exists(config.dataset_meta):
      downloader.download_and_extract(config.dataset_meta,
                                      config.dataset_url,
                                      False)
    else:
      print("Found " + config.dataset_meta + ".")

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
