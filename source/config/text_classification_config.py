from .config import Config


class TextClassificationCallbackConfig(Config):
  def __init__(self,
               default_callback_config):

    self.copy_props(default_callback_config)


class TextClassificationInputterConfig(Config):
  def __init__(self,
               default_inputter_config,
               vocab_file="",
               vocab_top_k=-1,
               encode_method=""):

    self.copy_props(default_inputter_config)
    self.vocab_file = vocab_file
    self.vocab_top_k = vocab_top_k
    self.encode_method = encode_method


class TextClassificationModelerConfig(Config):
  def __init__(self,
               default_modeler_config,
               num_classes=2,
               lr_method="step"):

    self.copy_props(default_modeler_config)
    self.num_classes = num_classes
    self.lr_method = lr_method
