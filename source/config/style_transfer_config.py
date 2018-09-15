from config import Config

class StyleTransferCallbackConfig(Config):
  def __init__(self,
               default_callback_config):

    self.copy_props(default_callback_config)


class StyleTransferInputterConfig(Config):
  def __init__(self,
               default_inputter_config,
               image_height=32,
               image_width=32,
               image_depth=3,
               resize_side_min=400,
               resize_side_max=600):

    self.copy_props(default_inputter_config)

    self.image_height = image_height
    self.image_width = image_width
    self.image_depth = image_depth    
    self.resize_side_min = resize_side_min
    self.resize_side_max = resize_side_max

class StyleTransferModelerConfig(Config):
  def __init__(self,
               default_modeler_config,
               data_format="channels_first",
               image_depth=3,
               style_weight=100.0,
               content_weight=7.5,
               tv_weight=200.0,
               feature_net="vgg_19_conv",
               feature_net_path=None,
               style_image_path=None):

    self.copy_props(default_modeler_config)
    self.data_format = data_format
    self.image_depth = image_depth
    self.style_weight = style_weight
    self.content_weight = content_weight
    self.tv_weight = tv_weight
    self.feature_net = feature_net
    self.feature_net_path = feature_net_path
    self.style_image_path = style_image_path