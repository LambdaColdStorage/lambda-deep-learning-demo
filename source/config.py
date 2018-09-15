
def copy_props(source_obj, target_obj):
  public_props = (
    name for name in dir(source_obj) if not name.startswith('_'))
  for props_name in public_props:
    setattr(target_obj, props_name, getattr(source_obj, props_name))


class Config(object):
  def __init__(self,
               mode="train",
               batch_size_per_gpu=10,
               gpu_count=1):

      self.mode = mode
      self.batch_size_per_gpu = batch_size_per_gpu
      self.gpu_count = gpu_count


class InputterConfig(object):
  def __init__(self,
               general_config,
               epochs=1,
               dataset_meta=None,
               test_samples=None,
               image_height=32,
               image_width=32,
               image_depth=3,
               data_format="channels_last",
               augmenter_speed_mode=False,
               shuffle_buffer_size=1000,
               num_classes=10):

    copy_props(general_config, self)
    self.epochs = epochs
    self.dataset_meta = dataset_meta
    self.test_samples = test_samples
    self.image_height = image_height
    self.image_width = image_width
    self.image_depth = image_depth
    self.augmenter_speed_mode = augmenter_speed_mode
    self.shuffle_buffer_size = shuffle_buffer_size
    self.data_format = data_format
    self.augmenter_speed_mode = augmenter_speed_mode
    self.shuffle_buffer_size = shuffle_buffer_size
    self.num_classes = num_classes


class ModelerConfig(object):
  def __init__(self,
               general_config,
               optimizer="momentum",
               learning_rate=0.1,
               trainable_vars=None,
               skip_trainable_vars=None,
               piecewise_boundaries=None,
               piecewise_lr_decay=None,
               skip_l2_loss_vars=None,
               num_classes=10,
               data_format="channels_first",
               l2_weight_decay=0.0002):

    copy_props(general_config, self)
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.trainable_vars = trainable_vars
    self.skip_trainable_vars = skip_trainable_vars
    self.piecewise_boundaries = piecewise_boundaries
    self.piecewise_lr_decay = piecewise_lr_decay
    self.skip_l2_loss_vars = skip_l2_loss_vars
    self.num_classes = num_classes
    self.data_format = data_format
    self.l2_weight_decay = l2_weight_decay


class RunnerConfig(object):
  def __init__(self,
               general_config,
               model_dir=None,
               summary_names=None,
               log_every_n_iter=10,
               save_summary_steps=100,
               pretrained_dir=None,
               skip_pretrained_var=None,
               save_checkpoints_steps=None,
               keep_checkpoint_max=5,
               train_callbacks=None,
               eval_callbacks=None,
               infer_callbacks=None):

    copy_props(general_config, self)
    self.model_dir = model_dir
    self.summary_names = summary_names
    self.log_every_n_iter = log_every_n_iter
    self.save_summary_steps = save_summary_steps
    self.pretrained_dir = pretrained_dir
    self.skip_pretrained_var = skip_pretrained_var
    self.save_checkpoints_steps = save_checkpoints_steps
    self.keep_checkpoint_max = keep_checkpoint_max
    self.train_callbacks = train_callbacks
    self.eval_callbacks = eval_callbacks
    self.infer_callbacks = infer_callbacks
