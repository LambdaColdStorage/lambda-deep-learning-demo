from .config import Config


class TextClassificationCallbackConfig(Config):
  def __init__(self,
               default_callback_config):

    self.copy_props(default_callback_config)


class TextClassificationInputterConfig(Config):
  def __init__(self,
               default_inputter_config,
               vocab_file=""):

    self.copy_props(default_inputter_config)
    self.vocab_file = vocab_file


class TextClassificationModelerConfig(Config):
  def __init__(self,
               default_modeler_config):

    self.copy_props(default_modeler_config)
