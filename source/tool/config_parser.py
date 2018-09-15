import yaml
import os


def parse(config_path):
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
    pretrained_dir=config.pretrained_dir,
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