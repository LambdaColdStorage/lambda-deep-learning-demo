import yaml
import os


def parse(config_path):
  """Parse a config file into a config object.
  """
  with open(config_path) as file:
    config = yaml.load(file.read())
  return config


def prepare(config):

  config.dataset_meta = os.path.expanduser(config.dataset_meta)
  config.model_dir = os.path.expanduser(config.model_dir)
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