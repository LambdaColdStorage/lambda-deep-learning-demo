from .config import Config


class TextGenerationCallbackConfig(Config):
  def __init__(self,
               default_callback_config,
               unit="char",
               softmax_temperature=1.0):

    self.copy_props(default_callback_config)
    self.unit = unit
    self.softmax_temperature = softmax_temperature

class TextGenerationInputterConfig(Config):
  def __init__(self,
               default_inputter_config,
               vocab_file="",
               vocab_top_k=-1,
               vocab_format="pickle",
               encode_method="",
               unit="char",
               starter="T"):

    self.copy_props(default_inputter_config)
    self.vocab_file = vocab_file
    self.vocab_top_k = vocab_top_k
    self.vocab_format = vocab_format
    self.encode_method = encode_method
    self.starter = starter
    self.unit = unit


class TextGenerationModelerConfig(Config):
  def __init__(self,
               default_modeler_config):

    self.copy_props(default_modeler_config)
