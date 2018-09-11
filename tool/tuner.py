import os

import app
import config_parser
import random

CONFIG_TUNE_PATH = "tool/config_tune.yaml"


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

      args.mode = "train"
      trainer = app.APP(args)
      trainer.run()

      args.mode = "eval"
      args.epochs = 1
      
      # Optional, set args.dataset_meta to a validate file
      # args.dataset_meta = path-to-eval-csv
      
      evaluator = app.APP(args)
      evaluator.run()
      t = t + 1