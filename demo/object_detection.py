"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

python demo/object_detection.py \
--mode=train --model_dir=~/demo/model/ssd512_mscoco \
--network=ssd512 --augmenter=ssd_augmenter --batch_size_per_gpu=16 --epochs=128 \
--dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=512 \
train_args --learning_rate=0.001 --optimizer=momentum --piecewise_boundaries=10000 \
--piecewise_lr_decay=1.0,0.1 --dataset_meta=train2017 \
--callbacks=train_basic,train_loss,train_speed,train_summary --trainable_vars=SSD \
--skip_l2_loss_vars=l2_norm_scaler --summary_names=loss,learning_rate,class_losses,bboxes_losses


CUDA_VISIBLE_DEVICES=0 python demo/object_detection.py \
--mode=train --model_dir=~/demo/model/ssd512_mscoco \
--network=ssd512 --augmenter=ssd_augmenter --batch_size_per_gpu=1 --epochs=128 \
--dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=512 \
train_args --learning_rate=0.001 --optimizer=momentum --piecewise_boundaries=10000 \
--piecewise_lr_decay=1.0,0.1 --dataset_meta=minival2014 \
--callbacks=train_basic,train_loss,train_speed,train_summary --trainable_vars=SSD \
--skip_l2_loss_vars=l2_norm_scaler --summary_names=loss,learning_rate,class_losses,bboxes_losses

python demo/object_detection.py \
--mode=eval \
--model_dir=~/demo/model/ssd512_mscoco \
--network=ssd512 \
--augmenter=ssd_augmenter \
--batch_size_per_gpu=8 --epochs=1 \
--dataset_dir=/mnt/data/data/mscoco \
eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco

CUDA_VISIBLE_DEVICES=0 python demo/object_detection.py --mode=infer --model_dir=~/ssd512_mscoco \
--network=ssd512 --augmenter=ssd_augmenter --batch_size_per_gpu=1 --epochs=1 \
--dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=512 \
infer_args --callbacks=infer_basic,infer_display_object_detection \
--test_samples=/mnt/data/data/mscoco/val2014/COCO_val2014_000000000042.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000073.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000074.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000133.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000136.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000143.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000164.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000192.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000196.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000208.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000241.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000257.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000283.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000294.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000328.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000338.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000357.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000359.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000360.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000387.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000395.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000397.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000400.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000415.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000428.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000459.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000472.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000474.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000486.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000488.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000502.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000520.jpg

python demo/object_detection.py \
--mode=tune \
--model_dir=~/demo/model/ssd512_mscoco \
--network=ssd512 \
--augmenter=ssd_augmenter \
--batch_size_per_gpu=16 \
--dataset_dir=/mnt/data/data/mscoco \
tune_args \
--train_callbacks=train_basic,train_loss,train_speed,train_summary \
--eval_callbacks=eval_basic,eval_speed,eval_mscoco \
--train_dataset_meta=train2017 \
--eval_dataset_meta=val2017 \
--tune_config=source/tool/ssd512_mscoco_tune_coarse.yaml \
--eval_reduce_ops=False \
--trainable_vars=SSD \
--skip_l2_loss_vars=l2_norm_scaler

"""
import sys
import os
import importlib


def main():

  sys.path.append('.')

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
  parser.add_argument("--resolution",
                      help="Image resolution used for detectoin.",
                      type=int,
                      default=512) 
  parser.add_argument("--dataset_dir",
                      help="Path to dataset.",
                      type=str,
                      default="/mnt/data/data/mscoco")
  parser.add_argument("--feature_net",
                      help="Name of feature net",
                      default="vgg_16_ssd512")
  parser.add_argument("--feature_net_path",
                      help="Path to pre-trained vgg model.",
                      default=os.path.join(
                        os.environ['HOME'],
                        "demo/model/vgg_16_2016_08_28/vgg_16.ckpt"))
  parser.add_argument("--data_format",
                      help="channels_first or channels_last",
                      choices=["channels_first", "channels_last"],
                      default="channels_last")
  config = parser.parse_args()

  config = config_parser.prepare(config)

  # Object detection can take a list of meta files (Other application should too)
  if hasattr(config, 'train_dataset_meta'):
    config.train_dataset_meta = (
      None if not config.train_dataset_meta
      else config.train_dataset_meta.split(","))

    if not isinstance(
      config.train_dataset_meta, (list, tuple)):
        config.train_dataset_meta = \
          [config.train_dataset_meta]

  if hasattr(config, 'eval_dataset_meta'):
    config.eval_dataset_meta = (
      None if not config.eval_dataset_meta
      else config.eval_dataset_meta.split(","))

    if not isinstance(
      config.eval_dataset_meta, (list, tuple)):
        config.eval_dataset_meta = \
          [config.eval_dataset_meta]

  # Generate config
  runner_config, callback_config, inputter_config, modeler_config = \
      config_parser.default_config(config)

  inputter_config = ObjectDetectionInputterConfig(
    inputter_config,
    dataset_dir=config.dataset_dir,
    num_classes=config.num_classes,
    resolution=config.resolution)

  modeler_config = ObjectDetectionModelerConfig(
    modeler_config,
    num_classes=config.num_classes,
    data_format=config.data_format,
    feature_net=config.feature_net,
    feature_net_path=config.feature_net_path)

  if config.mode == "tune":
    inputter_module = importlib.import_module(
      "source.inputter.object_detection_mscoco_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.object_detection_modeler")
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

    net = importlib.import_module(
      "source.network." + config.network)

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
