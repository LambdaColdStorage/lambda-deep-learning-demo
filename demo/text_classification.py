"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
"""
import sys
import os
import importlib

"""
Text Classification Demo
"""


def main():

  sys.path.append('.')

  from source.tool import downloader
  from source.tool import tuner
  from source.tool import config_parser

  from source.config.text_classification_config import \
      TextClassificationInputterConfig

  parser = config_parser.default_parser()
  app_parser = parser.add_argument_group('app')
  app_parser.add_argument("--vocab_file",
                          help="Path of the vocabulary file.",
                          type=str,
                          default="")
  # Default configs
  runner_config, callback_config, inputter_config, modeler_config, app_config = \
      config_parser.default_config(parser)

  inputter_config = TextClassificationInputterConfig(
    inputter_config,
    vocab_file=app_config.vocab_file)

  # Download data if necessary
  downloader.check_and_download(inputter_config)

  if runner_config.mode == "tune":

    inputter_module = importlib.import_module(
      "source.inputter.text_classification_inputter")
    modeler_module = importlib.import_module(
      "source.modeler.text_classification_modeler")
    runner_module = importlib.import_module(
      "source.runner.parameter_server_runner")

    tuner.tune(app_config,
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

    augmenter = (None if not inputter_config.augmenter else
                 importlib.import_module(
                  "source.augmenter." + inputter_config.augmenter))

    net = importlib.import_module(
      "source.network." + modeler_config.network)

    callbacks = []
    for name in callback_config.callbacks:
      callback = importlib.import_module(
        "source.callback." + name).build(callback_config)
      callbacks.append(callback)

    inputter = importlib.import_module(
      "source.inputter.text_classification_inputter").build(
      inputter_config, augmenter)

    modeler = importlib.import_module(
      "source.modeler.text_classification_modeler").build(
      modeler_config, net)

    # inputter = importlib.import_module(
    #   "source.inputter.text_classification_pretrain_inputter").build(
    #   inputter_config, augmenter)

    # modeler = importlib.import_module(
    #   "source.modeler.text_classification_pretrain_modeler").build(
    #   modeler_config, net)

    # inputter = importlib.import_module(
    #   "source.inputter.text_classification_inputter").build(
    #   inputter_config, augmenter)

    # modeler = importlib.import_module(
    #   "source.modeler.text_classification_modeler").build(
    #   modeler_config, net)

    runner = importlib.import_module(
      "source.runner.parameter_server_runner").build(
      runner_config, inputter, modeler, callbacks)

    # Run application
    runner.dev()


if __name__ == "__main__":
  main()
