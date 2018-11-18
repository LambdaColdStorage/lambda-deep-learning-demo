from config import Config

class ObjectDetectionCallbackConfig(Config):
  def __init__(self,
               default_callback_config,
               num_classes=81):

    self.copy_props(default_callback_config)
    self.num_classes = num_classes


class ObjectDetectionInputterConfig(Config):
  def __init__(self,
               default_inputter_config,
               dataset_dir="",
               num_classes=81,
               resolution=512):

    default_inputter_config.dataset_meta = (
      None if not default_inputter_config.dataset_meta
      else default_inputter_config.dataset_meta.split(","))

    if not isinstance(
      default_inputter_config.dataset_meta, (list, tuple)):
        default_inputter_config.dataset_meta = \
          [default_inputter_config.dataset_meta]

    self.copy_props(default_inputter_config)
    self.dataset_dir = dataset_dir
    self.num_classes = num_classes
    self.resolution = resolution


class ObjectDetectionModelerConfig(Config):
  def __init__(self,
               default_modeler_config,
               num_classes=81):

    self.copy_props(default_modeler_config)
    self.num_classes = num_classes