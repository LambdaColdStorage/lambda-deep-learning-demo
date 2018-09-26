from config import Config


class TextGenerationCallbackConfig(Config):
  def __init__(self,
               default_callback_config):

    self.copy_props(default_callback_config)


class TextGenerationInputterConfig(Config):
  def __init__(self,
               default_inputter_config):

    self.copy_props(default_inputter_config)


class TextGenerationModelerConfig(Config):
  def __init__(self,
               default_modeler_config):

    self.copy_props(default_modeler_config)
