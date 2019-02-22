import yaml
import os
import argparse

from tensorflow.python.client import device_lib


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_gpu_count():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def default_parser():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--mode", choices=["train", "eval", "infer", "tune", "export"],
                      type=str,
                      help="Choose a job mode from train, eval, infer and export.",
                      default="train")
  parser.add_argument("--model_dir",
                      help="Directory to save mode",
                      type=str,
                      default=None)
  parser.add_argument("--dataset_url",
                      help="URL for downloading data",
                      default="")
  parser.add_argument("--network", choices=["resnet32", "resnet50", "inception_v4", "nasnet_A_large",
                                            "fcn", "unet",
                                            "fns",
                                            "rnn_basic",
                                            "seq2label_basic", "seq2label_pretrain", "seq2label_bert",
                                            "ssd512", "ssd300"],
                      type=str,
                      help="Choose a network architecture",
                      default=None)
  parser.add_argument("--augmenter",
                      choices=["cifar_augmenter", "inception_augmenter", "vgg_augmenter",
                               "fcn_augmenter", "unet_augmenter",
                               "fns_augmenter",
                               "ssd_augmenter"],
                      type=str,
                      help="Name of the augmenter",
                      default=None)
  parser.add_argument("--batch_size_per_gpu",
                      help="Number of images on each GPU.",
                      type=int,
                      default=64)
  parser.add_argument("--gpu_count",
                      help="Number of GPUs.",
                      type=int,
                      default=get_gpu_count())
  parser.add_argument("--epochs",
                      help="Number of epochs.",
                      type=int,
                      default=5)

  subparsers = parser.add_subparsers(title='mode', dest='action')

  train_parser = subparsers.add_parser("train_args", help="Train help")
  train_parser.add_argument("--dataset_meta", type=str,
                            help="Path to dataset's meta file",
                            default=None)
  train_parser.add_argument("--learning_rate",
                            help="Initial learning rate in training.",
                            type=float,
                            default=0.5)
  train_parser.add_argument("--piecewise_boundaries",
                            help="Epochs to decay learning rate",
                            default="2")
  train_parser.add_argument("--piecewise_lr_decay",
                            help="Decay ratio for learning rate",
                            default="1.0,0.1")
  train_parser.add_argument("--optimizer",
                            help="Name of optimizer",
                            choices=["adadelta", "adagrad", "adam", "ftrl",
                                     "momentum", "rmsprop", "sgd",
                                     "custom"],
                            default="momentum")
  train_parser.add_argument("--log_every_n_iter",
                            help="Number of steps to log",
                            type=int,
                            default=10)
  train_parser.add_argument("--save_summary_steps",
                            help="Number of steps to save summary.",
                            type=int,
                            default=10)
  train_parser.add_argument("--save_checkpoints_steps",
                            help="Number of steps to save checkpoints",
                            type=int,
                            default=100)
  train_parser.add_argument("--keep_checkpoint_max",
                            help="Maximum number of checkpoints to save.",
                            type=int,
                            default=1)
  train_parser.add_argument("--summary_names",
                            help="A string of comma seperated names for summary",
                            type=str,
                            default="loss,accuracy,learning_rate")
  train_parser.add_argument("--pretrained_model",
                            help="Path to pretrained network for transfer learning.",
                            type=str,
                            default=None)
  train_parser.add_argument("--skip_pretrained_var",
                            help="Variables to skip in restoring from \
                                  pretrained model (for transfer learning).",
                            type=str,
                            default=None)
  train_parser.add_argument("--trainable_vars",
                            help="List of trainable Variables. \
                                 If None all variables in TRAINABLE_VARIABLES \
                                 will be trained.",
                            type=str,
                            default=None)
  train_parser.add_argument("--skip_l2_loss_vars",
                            help="List of blacklisted trainable Variables for L2 \
                                 regularization.",
                            type=str,
                            default="BatchNorm,preact,postnorm,gamma,beta")
  train_parser.add_argument("--l2_weight_decay",
                            help="Weight for l2 loss",
                            type=float,
                            default=0.0002)
  train_parser.add_argument("--callbacks",
                            help="List of callbacks in training.",
                            type=str,
                            default="train_basic,train_loss,train_accuracy,train_speed,train_summary")
  train_parser.add_argument("--reduce_ops",
                            help="Whether need to do a reduce on the results collected from multiple gpus",
                            type=str2bool,
                            default=True)

  eval_parser = subparsers.add_parser("eval_args", help="Eval help")
  eval_parser.add_argument("--dataset_meta", type=str,
                           help="Path to dataset's meta file",
                           default="")
  eval_parser.add_argument("--log_every_n_iter",
                           help="Number of steps to log",
                           type=int,
                           default=10)
  eval_parser.add_argument("--skip_l2_loss_vars",
                           help="List of blacklisted trainable Variables for L2 \
                                 regularization.",
                           type=str,
                           default="BatchNorm,preact,postnorm,gamma,beta")
  eval_parser.add_argument("--l2_weight_decay",
                           help="Weight for l2 loss",
                           type=float,
                           default=0.0002)
  eval_parser.add_argument("--callbacks",
                           help="List of callbacks in evaluation.",
                           type=str,
                           default="eval_basic,eval_loss,eval_accuracy,eval_speed,eval_summary")
  eval_parser.add_argument("--reduce_ops",
                           help="Whether need to do a reduce on the results collected from multiple gpus",
                           type=str2bool,
                           default=True)

  infer_parser = subparsers.add_parser("infer_args", help="Infer help")
  infer_parser.add_argument("--dataset_meta", type=str,
                            help="Path to dataset's meta file",
                            default="")  
  infer_parser.add_argument("--test_samples",
                            help="A string of comma seperated testing data. "
                            "Must be provided for infer mode.",
                            type=str,
                            default=None)
  infer_parser.add_argument("--callbacks",
                            help="List of callbacks in inference.",
                            type=str,
                            default=None)
  infer_parser.add_argument("--splitter",
                            help="A special character to split test_samples into a list",
                            type=str,
                            default=",")

  tune_parser = subparsers.add_parser("tune_args", help="Tune help")
  tune_parser.add_argument("--tune_config_path",
                           help="Config file for hyper-parameter tunning",
                           type=str,
                           default=None)
  tune_parser.add_argument("--train_dataset_meta", type=str,
                           help="Path to dataset's training meta file",
                           default=None)
  tune_parser.add_argument("--eval_dataset_meta", type=str,
                           help="Path to dataset's evaluation meta file",
                           default=None)
  tune_parser.add_argument("--learning_rate",
                           help="Initial learning rate in training.",
                           type=float,
                           default=0.5)
  tune_parser.add_argument("--piecewise_boundaries",
                           help="Epochs to decay learning rate",
                           default="2")
  tune_parser.add_argument("--piecewise_lr_decay",
                           help="Decay ratio for learning rate",
                           default="1.0,0.1")
  tune_parser.add_argument("--optimizer",
                           help="Name of optimizer",
                           choices=["adadelta", "adagrad", "adam", "ftrl",
                                    "momentum", "rmsprop", "sgd"],
                           default="momentum")
  tune_parser.add_argument("--log_every_n_iter",
                           help="Number of steps to log",
                           type=int,
                           default=10)
  tune_parser.add_argument("--save_summary_steps",
                           help="Number of steps to save summary.",
                           type=int,
                           default=10)
  tune_parser.add_argument("--save_checkpoints_steps",
                           help="Number of steps to save checkpoints",
                           type=int,
                           default=100)
  tune_parser.add_argument("--keep_checkpoint_max",
                           help="Maximum number of checkpoints to save.",
                           type=int,
                           default=1)
  tune_parser.add_argument("--summary_names",
                           help="A string of comma seperated names for summary",
                           type=str,
                           default="loss,accuracy,learning_rate")
  tune_parser.add_argument("--pretrained_model",
                           help="Path to pretrained network for transfer learning.",
                           type=str,
                           default=None)
  tune_parser.add_argument("--skip_pretrained_var",
                           help="Variables to skip in restoring from \
                                 pretrained model (for transfer learning).",
                           type=str,
                           default=None)
  tune_parser.add_argument("--trainable_vars",
                           help="List of trainable Variables. \
                                If None all variables in TRAINABLE_VARIABLES \
                                 will be trained, subjected to the ones.",
                           type=str,
                           default=None)
  tune_parser.add_argument("--skip_l2_loss_vars",
                           help="List of blacklisted trainable Variables for L2 \
                                regularization.",
                           type=str,
                           default="BatchNorm,preact,postnorm,gamma,beta")
  tune_parser.add_argument("--l2_weight_decay",
                           help="Weight for l2 loss",
                           type=float,
                           default=0.0002)
  tune_parser.add_argument("--train_callbacks",
                           help="List of callbacks in training.",
                           type=str,
                           default="train_basic,train_loss,train_accuracy,train_speed,train_summary")
  tune_parser.add_argument("--eval_callbacks",
                           help="List of callbacks in evaluation.",
                           type=str,
                           default="eval_basic,eval_loss,eval_accuracy,eval_speed,eval_summary")
  tune_parser.add_argument("--train_reduce_ops",
                           help="Whether need to do a reduce on the results collected from multiple gpus",
                           type=str2bool,
                           default=True)
  tune_parser.add_argument("--eval_reduce_ops",
                           help="Whether need to do a reduce on the results collected from multiple gpus",
                           type=str2bool,
                           default=True)

  export_parser = subparsers.add_parser("export_args", help="Export help")
  export_parser.add_argument("--callbacks",
                             help="List of callbacks in export.",
                             type=str,
                             default="export_basic")
  export_parser.add_argument("--export_dir",
                      help="Directory to export mode",
                      type=str,
                      default=None)
  export_parser.add_argument("--export_version",
                      help="Mode version",
                      type=str,
                      default="1")
  export_parser.add_argument("--input_ops",
                             help="list of input operations for exporting the model",
                             type=str,
                             default="")
  export_parser.add_argument("--output_ops",
                             help="list of output operations for exporting the model",
                             type=str,
                             default="")
  export_parser.add_argument("--dataset_meta", type=str,
                             help="Path to dataset's meta file",
                             default="")
  
  return parser


def yaml_parse(config_path):
  """Parse a config file into a config object.
  """
  with open(config_path) as file:
    config = yaml.load(file.read())
  return config


def prepare(config):

  if hasattr(config, 'dataset_meta'):
    config.dataset_meta = (
      None if not config.dataset_meta
      else config.dataset_meta.split(","))
    if not isinstance(
      config.dataset_meta, (list, tuple)):
        config.dataset_meta = \
          [config.dataset_meta]
    config.dataset_meta = [
      (None if not meta else os.path.expanduser(meta)) 
      for meta in config.dataset_meta]

  if hasattr(config, 'train_dataset_meta'):
    config.train_dataset_meta = (
      None if not config.train_dataset_meta
      else config.train_dataset_meta.split(","))
    if not isinstance(
      config.train_dataset_meta, (list, tuple)):
        config.train_dataset_meta = \
          [config.train_dataset_meta]
    config.train_dataset_meta = [
      (None if not meta else os.path.expanduser(meta)) 
      for meta in config.train_dataset_meta]

  if hasattr(config, 'eval_dataset_meta'):
    config.eval_dataset_meta = (
      None if not config.eval_dataset_meta
      else config.eval_dataset_meta.split(","))
    if not isinstance(
      config.eval_dataset_meta, (list, tuple)):
        config.eval_dataset_meta = \
          [config.eval_dataset_meta]
    config.eval_dataset_meta = [
      (None if not meta else os.path.expanduser(meta)) 
      for meta in config.eval_dataset_meta]


  if hasattr(config, "model_dir"):
    config.model_dir = ("" if not config.model_dir else
                        os.path.expanduser(config.model_dir))

  if hasattr(config, "summary_names"):
    config.summary_names = (
      [] if not config.summary_names else
      config.summary_names.split(","))

  if hasattr(config, "skip_pretrained_var"):
    config.skip_pretrained_var = (
      [] if not config.skip_pretrained_var else
      config.skip_pretrained_var.split(","))

  if hasattr(config, "trainable_vars"):
    config.trainable_vars = (
      [] if not config.trainable_vars else
      config.trainable_vars.split(","))

  if hasattr(config, "skip_l2_loss_vars"):
    config.skip_l2_loss_vars = (
      [] if not config.skip_l2_loss_vars else
      config.skip_l2_loss_vars.split(","))

  if hasattr(config, "augmenter"):
    config.augmenter = (
      None if not config.augmenter else config.augmenter)

  if hasattr(config, "piecewise_boundaries"):
    config.piecewise_boundaries = (
      [] if not config.piecewise_boundaries else
      list(map(float, config.piecewise_boundaries.split(","))))

  if hasattr(config, "piecewise_lr_decay"):
    config.piecewise_lr_decay = (
      [] if not config.piecewise_lr_decay else
      list(map(float, config.piecewise_lr_decay.split(","))))

  if hasattr(config, "test_samples"):
    config.test_samples = (
      [] if not config.test_samples else
      [os.path.expanduser(x) for x in config.test_samples.split(config.splitter)])

  if hasattr(config, "callbacks"):
    config.callbacks = (
      [] if not config.callbacks else
      config.callbacks.split(","))

  if hasattr(config, "train_callbacks"):
    config.train_callbacks = (
      [] if not config.train_callbacks else
      config.train_callbacks.split(","))

  if hasattr(config, "eval_callbacks"):
    config.eval_callbacks = (
      [] if not config.eval_callbacks else
      config.eval_callbacks.split(","))

  if hasattr(config, "input_ops"):
    config.input_ops = (
      [] if not config.input_ops else
      config.input_ops.split(","))

  if hasattr(config, "output_ops"):
    config.output_ops = (
      [] if not config.output_ops else
      config.output_ops.split(","))

  return config


def default_config(parser):

  import sys
  sys.path.append('.')

  from source.config.config import (RunnerConfig, CallbackConfig,
                                    InputterConfig, ModelerConfig)

  args = parser.parse_args()

  config = prepare(args)

  # Create configs
  runner_config = RunnerConfig(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count,
    summary_names=(None if not hasattr(config, "summary_names")
                   else config.summary_names),
    reduce_ops=(True if not hasattr(config, "reduce_ops")
                else config.reduce_ops),
    train_reduce_ops=(True if not hasattr(config, "train_reduce_ops")
                else config.train_reduce_ops),
    eval_reduce_ops=(True if not hasattr(config, "eval_reduce_ops")
                else config.eval_reduce_ops))

  callback_config = CallbackConfig(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count,
    model_dir=config.model_dir,
    log_every_n_iter=(None if not hasattr(config, "log_every_n_iter")
                      else config.log_every_n_iter),
    save_summary_steps=(None if not hasattr(config, "save_summary_steps")
                        else config.save_summary_steps),
    pretrained_model=(None if not hasattr(config, "pretrained_model")
                      else config.pretrained_model),
    skip_pretrained_var=(None if not hasattr(config, "skip_pretrained_var")
                         else config.skip_pretrained_var),
    save_checkpoints_steps=(None if not hasattr(config, "save_checkpoints_steps")
                            else config.save_checkpoints_steps),
    keep_checkpoint_max=(None if not hasattr(config, "keep_checkpoint_max")
                         else config.keep_checkpoint_max),
    callbacks=(None if not hasattr(config, "callbacks")
                         else config.callbacks),
    train_callbacks=(None if not hasattr(config, "train_callbacks")
                         else config.train_callbacks),
    eval_callbacks=(None if not hasattr(config, "eval_callbacks")
                         else config.eval_callbacks),
    export_dir=(None if not hasattr(config, "export_dir")
                     else config.export_dir),
    export_version=(None if not hasattr(config, "export_version")
                         else config.export_version),
    input_ops=(None if not hasattr(config, "input_ops")
                    else config.input_ops),
    output_ops=(None if not hasattr(config, "output_ops")
                    else config.output_ops)
    )


  inputter_config = InputterConfig(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count,
    epochs=config.epochs,
    dataset_url=(None if not hasattr(config, "dataset_url")
                  else config.dataset_url),
    dataset_meta=(None if not hasattr(config, "dataset_meta")
                  else config.dataset_meta),
    train_dataset_meta=(None if not hasattr(config, "train_dataset_meta")
                        else config.train_dataset_meta),
    eval_dataset_meta=(None if not hasattr(config, "eval_dataset_meta")
                       else config.eval_dataset_meta),
    test_samples=(None if not hasattr(config, "test_samples")
                  else config.test_samples),
    augmenter=(None if not hasattr(config, "augmenter")
               else config.augmenter),
    augmenter_speed_mode=(None if not hasattr(config, "augmenter_speed_mode")
                          else config.augmenter_speed_mode))


  modeler_config = ModelerConfig(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count,
    optimizer=(None if not hasattr(config, "optimizer")
               else config.optimizer),
    learning_rate=(None if not hasattr(config, "learning_rate")
                   else config.learning_rate),
    trainable_vars=(None if not hasattr(config, "trainable_vars")
                    else config.trainable_vars),
    piecewise_boundaries=(None if not hasattr(config, "piecewise_boundaries")
                          else config.piecewise_boundaries),
    piecewise_lr_decay=(None if not hasattr(config, "piecewise_lr_decay")
                        else config.piecewise_lr_decay),
    skip_l2_loss_vars=(None if not hasattr(config, "skip_l2_loss_vars")
                       else config.skip_l2_loss_vars),
    l2_weight_decay=(None if not hasattr(config, "l2_weight_decay")
                     else config.l2_weight_decay),
    network=(None if not hasattr(config, "network")
                     else config.network),
    tune_config_path=(None if not hasattr(config, "tune_config_path")
                     else config.tune_config_path))


  arg_groups={}
  for group in parser._action_groups:
      group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
      arg_groups[group.title]=argparse.Namespace(**group_dict)
  app_config = arg_groups['app']

  return runner_config, callback_config, inputter_config, modeler_config, app_config