class Config(object):
  def __init__(self,
               mode,
               batch_size_per_gpu,
               gpu_count):

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
               mode,
               batch_size_per_gpu,
               gpu_count,
               summary_names,
               reduce_ops,
               train_reduce_ops,
               eval_reduce_ops):
    super(RunnerConfig, self).__init__(
      mode, batch_size_per_gpu, gpu_count)

    self.summary_names = summary_names
    self.reduce_ops = reduce_ops
    self.train_reduce_ops = train_reduce_ops
    self.eval_reduce_ops = eval_reduce_ops


class CallbackConfig(Config):
  def __init__(self,
               mode,
               batch_size_per_gpu,
               gpu_count,
               model_dir,
               log_every_n_iter,
               save_summary_steps,
               pretrained_model,
               skip_pretrained_var,
               save_checkpoints_steps,
               keep_checkpoint_max,
               callbacks,
               train_callbacks,
               eval_callbacks,
               export_dir,
               export_version,
               input_ops,
               output_ops):

    super(CallbackConfig, self).__init__(
      mode, batch_size_per_gpu, gpu_count)

    self.model_dir = model_dir
    self.log_every_n_iter = log_every_n_iter
    self.save_summary_steps = save_summary_steps
    self.pretrained_model = pretrained_model
    self.skip_pretrained_var = skip_pretrained_var
    self.save_checkpoints_steps = save_checkpoints_steps
    self.keep_checkpoint_max = keep_checkpoint_max
    self.callbacks = callbacks
    self.train_callbacks = train_callbacks
    self.eval_callbacks = eval_callbacks
    self.export_dir = export_dir
    self.export_version = export_version
    self.input_ops = input_ops
    self.output_ops = output_ops


class InputterConfig(Config):
  def __init__(self,
               mode,
               batch_size_per_gpu,
               gpu_count,
               epochs,
               dataset_meta,
               train_dataset_meta,
               eval_dataset_meta,
               test_samples,
               augmenter,
               augmenter_speed_mode):

    super(InputterConfig, self).__init__(
      mode, batch_size_per_gpu, gpu_count)

    self.epochs = epochs
    self.dataset_meta = dataset_meta
    self.train_dataset_meta = train_dataset_meta
    self.eval_dataset_meta = eval_dataset_meta
    self.test_samples = test_samples
    self.augmenter = augmenter
    self.augmenter_speed_mode = augmenter_speed_mode


class ModelerConfig(Config):
  def __init__(self,
               mode,
               batch_size_per_gpu,
               gpu_count,
               optimizer,
               learning_rate,
               trainable_vars,
               piecewise_boundaries,
               piecewise_lr_decay,
               skip_l2_loss_vars,
               l2_weight_decay,
               network,
               tune_config_path):

    super(ModelerConfig, self).__init__(
      mode, batch_size_per_gpu, gpu_count)

    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.trainable_vars = trainable_vars
    self.piecewise_boundaries = piecewise_boundaries
    self.piecewise_lr_decay = piecewise_lr_decay
    self.skip_l2_loss_vars = skip_l2_loss_vars
    self.l2_weight_decay = l2_weight_decay
    self.network = network
    self.tune_config_path = tune_config_path
