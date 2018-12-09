from .config import Config

class ImageClassificationCallbackConfig(Config):
  def __init__(self,
               default_callback_config):

    self.copy_props(default_callback_config)


class ImageClassificationInputterConfig(Config):
  def __init__(self,
               default_inputter_config,
               image_height=32,
               image_width=32,
               image_depth=3,               
               num_classes=10):

    self.copy_props(default_inputter_config)

    self.image_height = image_height
    self.image_width = image_width
    self.image_depth = image_depth    
    self.num_classes = num_classes


class ImageClassificationModelerConfig(Config):
  def __init__(self,
               default_modeler_config,
               num_classes=10,
               data_format="channels_first"):

    self.copy_props(default_modeler_config)
    self.num_classes = num_classes
    self.data_format = data_format