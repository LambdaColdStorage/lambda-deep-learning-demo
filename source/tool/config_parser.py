import yaml
import os
import argparse


def default_parser():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
                      default=128)
  parser.add_argument("--gpu_count",
                      help="Number of GPUs.",
                      type=int,
                      default=4)
  parser.add_argument("--epochs",
                      help="Number of epochs.",
                      type=int,
                      default=5)
  parser.add_argument("--model_dir",
                      help="Directory to save mode",
                      type=str,
                      default="")
  parser.add_argument("--learning_rate",
                      help="Initial learning rate in training.",
                      type=float,
                      default=0.5)
  parser.add_argument("--piecewise_boundaries",
                      help="Epochs to decay learning rate",
                      default="2")
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
  parser.add_argument("--test_samples",
                      help="A string of comma seperated testing data. "
                      "Must be provided for infer mode.",
                      type=str)
  parser.add_argument("--summary_names",
                      help="A string of comma seperated names for summary",
                      type=str,
                      default="loss,accuracy,learning_rate")
  parser.add_argument("--pretrained_model",
                      help="Path to pretrained network for transfer learning.",
                      type=str,
                      default="")
  parser.add_argument("--skip_pretrained_var",
                      help="Variables to skip in restoring from \
                            pretrained model (for transfer learning).",
                      type=str,
                      default="")
  parser.add_argument("--trainable_vars",
                      help="List of trainable Variables. \
                           If None all variables in TRAINABLE_VARIABLES \
                           will be trained, subjected to the ones \
                           blacklisted by skip_trainable_vars.",
                      type=str,
                      default="")
  parser.add_argument("--skip_trainable_vars",
                      help="List of blacklisted trainable Variables.",
                      type=str,
                      default="")
  parser.add_argument("--skip_l2_loss_vars",
                      help="List of blacklisted trainable Variables for L2 \
                            regularization.",
                      type=str,
                      default="BatchNorm,preact,postnorm")
  parser.add_argument("--tune_config_path",
                      help="Config file for hyper-parameter tunning",
                      type=str,
                      default="")

  return parser


def yaml_parse(config_path):
  """Parse a config file into a config object.
  """
  with open(config_path) as file:
    config = yaml.load(file.read())
  return config


def prepare(config):

  config.dataset_meta = ("" if not config.dataset_meta else
    os.path.expanduser(config.dataset_meta))

  config.model_dir = ("" if not config.model_dir else
    os.path.expanduser(config.model_dir))

  config.summary_names = (
    [] if not config.summary_names else
    config.summary_names.split(","))

  config.skip_pretrained_var = (
    [] if not config.skip_pretrained_var else
    config.skip_pretrained_var.split(","))

  config.skip_trainable_vars = (
    [] if not config.skip_trainable_vars else
    config.skip_trainable_vars.split(","))

  config.trainable_vars = (
    [] if not config.trainable_vars else
    config.trainable_vars.split(","))

  config.skip_l2_loss_vars = (
    [] if not config.skip_l2_loss_vars else
    config.skip_l2_loss_vars.split(","))

  config.augmenter = (
    None if not config.augmenter else config.augmenter)

  config.piecewise_boundaries = (
    [] if not config.piecewise_boundaries else
    list(map(float, config.piecewise_boundaries.split(","))))

  config.piecewise_lr_decay = (
    [] if not config.piecewise_lr_decay else
    list(map(float, config.piecewise_lr_decay.split(","))))

  config.test_samples = (
    [] if not config.test_samples else
    [os.path.expanduser(x) for x in config.test_samples.split(",")])

  config.train_callbacks = (
    [] if not config.train_callbacks else
    config.train_callbacks.split(","))

  config.eval_callbacks = (
    [] if not config.eval_callbacks else
    config.eval_callbacks.split(","))

  config.infer_callbacks = (
    [] if not config.infer_callbacks else
    config.infer_callbacks.split(","))

  return config


def default_config(config):

  import sys
  sys.path.append('.')

  from source.config.config import RunnerConfig, CallbackConfig, InputterConfig, ModelerConfig

  # Create configs
  runner_config = RunnerConfig(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count,    
    summary_names=config.summary_names)

  callback_config = CallbackConfig(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count,    
    model_dir=config.model_dir,
    log_every_n_iter=config.log_every_n_iter,
    save_summary_steps=config.save_summary_steps,
    pretrained_model=config.pretrained_model,
    skip_pretrained_var=config.skip_pretrained_var,
    save_checkpoints_steps=config.save_checkpoints_steps,
    keep_checkpoint_max=config.keep_checkpoint_max)

  inputter_config = InputterConfig(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count,    
    epochs=config.epochs,
    dataset_meta=config.dataset_meta,
    test_samples=config.test_samples)

  modeler_config = ModelerConfig(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count,    
    optimizer=config.optimizer,
    learning_rate=config.learning_rate,
    trainable_vars=config.trainable_vars,
    skip_trainable_vars=config.skip_trainable_vars,
    piecewise_boundaries=config.piecewise_boundaries,
    piecewise_lr_decay=config.piecewise_lr_decay,
    skip_l2_loss_vars=config.skip_l2_loss_vars)

  return runner_config, callback_config, inputter_config, modeler_config