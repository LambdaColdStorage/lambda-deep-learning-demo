from config import Config

class ObjectDetectionCallbackConfig(Config):
  def __init__(self,
               default_callback_config):

    self.copy_props(default_callback_config)


class ObjectDetectionInputterConfig(Config):
  def __init__(self,
               default_inputter_config,
               dataset_dir=""):

    default_inputter_config.dataset_meta = (
      None if not default_inputter_config.dataset_meta
      else default_inputter_config.dataset_meta.split(","))

    if not isinstance(
      default_inputter_config.dataset_meta, (list, tuple)):
        default_inputter_config.dataset_meta = \
          [default_inputter_config.dataset_meta]

    self.copy_props(default_inputter_config)
    self.dataset_dir = dataset_dir


class ObjectDetectionModelerConfig(Config):
  def __init__(self,
               default_modeler_config):

    self.copy_props(default_modeler_config)