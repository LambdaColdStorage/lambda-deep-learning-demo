class Config(object):
  def __init__(self,
               mode="train",
               batch_size_per_gpu=10,
               gpu_count=1):

      self.mode = mode
      self.batch_size_per_gpu = batch_size_per_gpu
      self.gpu_count = gpu_count

  def copy_props(self, source_obj):
    public_props = (
      name for name in dir(source_obj) if not name.startswith('_'))
    for props_name in public_props:
      setattr(self, props_name, getattr(source_obj, props_name))

class RunnerConfig(Config):
  def __init__(self,
               mode="train",
               batch_size_per_gpu=10,
               gpu_count=1,    
               summary_names=None):
    super(RunnerConfig, self).__init__(
      mode, batch_size_per_gpu, gpu_count)

    self.summary_names = summary_names


class CallbackConfig(Config):
  def __init__(self,
               mode="train",
               batch_size_per_gpu=10,
               gpu_count=1,    
               model_dir=None,
               log_every_n_iter=10,
               save_summary_steps=100,
               pretrained_model=None,
               skip_pretrained_var=None,
               save_checkpoints_steps=None,
               keep_checkpoint_max=5):

    super(CallbackConfig, self).__init__(
      mode, batch_size_per_gpu, gpu_count)

    self.model_dir = model_dir
    self.log_every_n_iter = log_every_n_iter
    self.save_summary_steps = save_summary_steps
    self.pretrained_model = pretrained_model
    self.skip_pretrained_var = skip_pretrained_var
    self.save_checkpoints_steps = save_checkpoints_steps
    self.keep_checkpoint_max = keep_checkpoint_max


class InputterConfig(Config):
  def __init__(self,
               mode="train",
               batch_size_per_gpu=10,
               gpu_count=1,    
               epochs=1,
               dataset_meta=None,
               test_samples=None,
               augmenter_speed_mode=False,
               shuffle_buffer_size=256):

    super(InputterConfig, self).__init__(
      mode, batch_size_per_gpu, gpu_count)

    self.epochs = epochs
    self.dataset_meta = dataset_meta
    self.test_samples = test_samples
    self.augmenter_speed_mode = augmenter_speed_mode
    self.shuffle_buffer_size = shuffle_buffer_size


class ModelerConfig(Config):
  def __init__(self,
               mode="train",
               batch_size_per_gpu=10,
               gpu_count=1,    
               optimizer="momentum",
               learning_rate=0.1,
               trainable_vars=None,
               skip_trainable_vars=None,
               piecewise_boundaries=None,
               piecewise_lr_decay=None,
               skip_l2_loss_vars=None,
               l2_weight_decay=0.0002):

    super(ModelerConfig, self).__init__(
      mode, batch_size_per_gpu, gpu_count)

    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.trainable_vars = trainable_vars
    self.skip_trainable_vars = skip_trainable_vars
    self.piecewise_boundaries = piecewise_boundaries
    self.piecewise_lr_decay = piecewise_lr_decay
    self.skip_l2_loss_vars = skip_l2_loss_vars
    self.l2_weight_decay = l2_weight_decay
