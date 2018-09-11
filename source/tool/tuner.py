import os
import random
import importlib

from source import app
from source.tool import config_parser


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


def train(args):
  args.mode = "train"

  # Create components of the application
  augmenter = (None if not args.augmenter else
               importlib.import_module(
                "source.augmenter." + args.augmenter))

  net = getattr(importlib.import_module(
    "source.network." + args.network), "net")

  if args.mode == "train":
    callback_names = args.train_callbacks.split(",")
  elif args.mode == "eval":
    callback_names = args.eval_callbacks.split(",")
  elif args.mode == "infer":
    callback_names = args.infer_callbacks.split(",")

  callbacks = []
  for name in callback_names:
    callback = importlib.import_module(
      "source.callback." + name).build(args)
    callbacks.append(callback)

  inputter = importlib.import_module(
    "source.inputter." + args.inputter).build(args, augmenter)

  modeler = importlib.import_module(
    "source.modeler." + args.modeler).build(args, net, callbacks)

  runner = importlib.import_module(
    "source.runner." + args.runner).build(args, inputter, modeler)

  trainer = app.APP(args, runner)
  trainer.run()


def eval(args):
  args.mode = "eval"
  args.epochs = 1

  # Create components of the application
  augmenter = (None if not args.augmenter else
               importlib.import_module(
                "source.augmenter." + args.augmenter))

  net = getattr(importlib.import_module(
    "source.network." + args.network), "net")

  if args.mode == "train":
    callback_names = args.train_callbacks.split(",")
  elif args.mode == "eval":
    callback_names = args.eval_callbacks.split(",")
  elif args.mode == "infer":
    callback_names = args.infer_callbacks.split(",")

  callbacks = []
  for name in callback_names:
    callback = importlib.import_module(
      "source.callback." + name).build(args)
    callbacks.append(callback)

  inputter = importlib.import_module(
    "source.inputter." + args.inputter).build(args, augmenter)

  modeler = importlib.import_module(
    "source.modeler." + args.modeler).build(args, net, callbacks)

  runner = importlib.import_module(
    "source.runner." + args.runner).build(args, inputter, modeler)

  # Optional, set args.dataset_meta to a validate file
  # args.dataset_meta = path-to-eval-csv

  evaluator = app.APP(args, runner)
  evaluator.run()


def tune(args):
  # Parse config file
  config = config_parser.parse(CONFIG_TUNE_PATH)

  # Setup the tuning jobs
  num_trials = config["num_trials"]

  dir_ori = os.path.join(args.model_dir, "tune", "trial")
  t = 0
  while t < num_trials:
    dir_update = dir_ori

    # Update fixed params (epochs needs to be reset)
    for field in config["fixedparams"].keys():
      setattr(args, field, config["fixedparams"][field])

    # Update hyper parameter
    for field in config["hyperparams"].keys():
      if hasattr(args, field):
        values = config["hyperparams"][field].split(",")
        v = random.choice(values)
        setattr(args, field, type_convert(v))
        dir_update = dir_update + "_" + field + "_" + str(v)

    if not os.path.isdir(dir_update):
      args.model_dir = dir_update

      train(args)

      eval(args)

      t = t + 1
