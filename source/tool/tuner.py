import os
import random
import importlib

from source.tool import config_parser
from source.config import Config
from source.config import InputterConfig
from source.config import ModelerConfig
from source.config import RunnerConfig


CONFIG_TUNE_PATH = "source/tool/config_tune.yaml"


def type_convert(v):
    """ convert value to int, float or str"""
    try:
        float(v)
        tp = 1 if v.count(".") == 0 else 2
    except ValueError:
        tp = -1
    if tp == 1:
      return int(v)
    elif tp == 2:
      return float(v)
    elif tp == -1:
      return v
    else:
      assert False, "Unknown type for hyper parameter: {}".format(tp)


def excute(config):
  # Create configs
  general_config = Config(
    mode=config.mode,
    batch_size_per_gpu=config.batch_size_per_gpu,
    gpu_count=config.gpu_count)

  inputter_config = InputterConfig(
    general_config,
    epochs=config.epochs,
    dataset_meta=config.dataset_meta,
    test_samples=config.test_samples,
    image_height=config.image_height,
    image_width=config.image_width,
    image_depth=config.image_depth,
    num_classes=config.num_classes)

  modeler_config = ModelerConfig(
    general_config,
    optimizer=config.optimizer,
    learning_rate=config.learning_rate,
    trainable_vars=config.trainable_vars,
    skip_trainable_vars=config.skip_trainable_vars,
    piecewise_boundaries=config.piecewise_boundaries,
    piecewise_lr_decay=config.piecewise_lr_decay,
    skip_l2_loss_vars=config.skip_l2_loss_vars,
    num_classes=config.num_classes)

  runner_config = RunnerConfig(
    general_config,
    model_dir=config.model_dir,
    summary_names=config.summary_names,
    log_every_n_iter=config.log_every_n_iter,
    save_summary_steps=config.save_summary_steps,
    pretrained_dir=config.pretrained_dir,
    skip_pretrained_var=config.skip_pretrained_var,
    save_checkpoints_steps=config.save_checkpoints_steps,
    keep_checkpoint_max=config.keep_checkpoint_max,
    train_callbacks=config.train_callbacks,
    eval_callbacks=config.eval_callbacks,
    infer_callbacks=config.infer_callbacks)

  callback_config = runner_config

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
    "source.inputter." + config.inputter).build(inputter_config, augmenter)

  modeler = importlib.import_module(
    "source.modeler." + config.modeler).build(modeler_config, net)

  runner = importlib.import_module(
    "source.runner." + config.runner).build(
    runner_config, inputter, modeler, callbacks)

  # Run application
  runner.run()


def train(config):
  config.mode = "train"

  excute(config)


def eval(config):
  config.mode = "eval"
  config.epochs = 1

  excute(config)


def tune(config):
  # Parse config file
  config = config_parser.parse(CONFIG_TUNE_PATH)

  # Setup the tuning jobs
  num_trials = config["num_trials"]

  dir_ori = os.path.join(config.model_dir, "tune", "trial")
  t = 0
  while t < num_trials:
    dir_update = dir_ori

    # Update fixed params (epochs needs to be reset)
    for field in config["fixedparams"].keys():
      setattr(config, field, config["fixedparams"][field])

    # Update hyper parameter
    for field in config["hyperparams"].keys():
      if hasattr(config, field):
        values = config["hyperparams"][field].split(",")
        v = random.choice(values)
        setattr(config, field, type_convert(v))
        dir_update = dir_update + "_" + field + "_" + str(v)

    if not os.path.isdir(dir_update):
      config.model_dir = dir_update

      train(config)

      eval(config)

      t = t + 1
