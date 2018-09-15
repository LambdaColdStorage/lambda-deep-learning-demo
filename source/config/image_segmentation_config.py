from config import Config

class ImageSegmentationCallbackConfig(Config):
  def __init__(self,
               callback_config,
               num_classes=10):

    copy_props(callback_config)
    self.num_classes = num_classes


class ImageSegmentationInputterConfig(Config):
  def __init__(self,
               inputter_config,
               image_height=32,
               image_width=32,
               image_depth=3,
               output_height=32,
               output_width=32,
               resize_side_min=32,
               resize_side_max=32,               
               num_classes=10):

    copy_props(inputter_config)
    self.image_height = image_height
    self.image_width = image_width
    self.image_depth = image_depth
    self.output_height = output_height
    self.output_width = output_width
    self.resize_side_min = resize_side_min
    self.resize_side_max = resize_side_max
    self.num_classes = num_classes


class ImageSegmentationModelerConfig(Config):
  def __init__(self,
               modeler_config,
               num_classes=10,
               data_format="channels_first"):

    self.copy_props(modeler_config)
    self.num_classes = num_classes
    self.data_format = data_format