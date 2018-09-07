"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from runner import Runner


class ParameterServerRunner(Runner):
  def __init__(self, args, inputter, modeler):
    super(ParameterServerRunner, self).__init__(args,
                                                inputter,
                                                modeler)
    self.ps_ops = ["Variable", "VariableV2", "AutoReloadVariable"]
    self.session_config = self.create_session_config()
    self.sess = None
    self.num_samples = inputter.get_num_samples()
    self.batch_size = self.args.batch_size_per_gpu * self.args.num_gpu
    self.modeler.num_samples = self.num_samples
    self.feed_dict = {}

  def create_session_config(self):
    """create session_config
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,
                                allow_growth=True)

    # set number of GPU devices
    device_count = {"GPU": self.args.num_gpu}

    session_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      device_count=device_count,
      gpu_options=gpu_options)

    return session_config

  def assign_to_device(self, device, ps_device="/cpu:0"):
      def _assign(op):
          node_def = op if isinstance(op, tf.NodeDef) else op.node_def
          if node_def.op in self.ps_ops:
              return "/" + ps_device
          else:
              return device
      return _assign

  def batch_split(self, batch, idx):
    bs_per_gpu = self.args.batch_size_per_gpu
    batch_per_gpu = ()
    for x in batch:
      batch_per_gpu = (batch_per_gpu +
                       (x[idx * bs_per_gpu:(idx + 1) * bs_per_gpu],))
    return batch_per_gpu

  def average_gradients(self, tower_grads):
    average_grads = []

    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        if g is not None:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a "tower" dimension which we will average over below.
          grads.append(expanded_g)

      if grads:
        # Average over the "tower" dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So we will just return the first tower"s pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

  def reduce_op(self, x):
    if isinstance(x[0], list):
      return self.average_gradients(x)
    else:
      return tf.reduce_mean(x)

  def replicate_graph(self, pre_fns, input_fn, model_fn):
    with tf.device("/cpu:0"):

      for fn in pre_fns:
        fn()

      if self.args.mode == "infer":
        pass
      else:
        batch = input_fn()
        output = {}
        # Map
        for i in range(self.args.num_gpu):
          with tf.device(self.assign_to_device("/gpu:{}".format(i),
                         ps_device="/cpu:0")):
            # Split input data across multiple devices
            x = self.batch_split(batch, i)
            y = model_fn(x)

            # Gather output across multiple devices
            if i == 0:
              for key in y:
                output[key] = [y[key]]
            else:
              for key in y:
                output[key].append(y[key])

        # Reduce
        reduced_ops = {}
        for key in output:
          reduced_ops[key] = self.reduce_op(output[key])
        return reduced_ops

  def create_graph(self):
    reduced_ops = self.replicate_graph(
      [self.modeler.create_precomputation,
       self.inputter.create_precomputation],
      self.inputter.input_fn,
      self.modeler.model_fn)

    # Create train_op for gradient, keep other ops unchanged
    self.run_ops = []
    self.name_ops = []
    for key in reduced_ops:
      if key == "grads":
        minimize_op = self.modeler.optimizer.apply_gradients(
          reduced_ops[key], global_step=self.modeler.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        op = tf.group(minimize_op, update_ops)
      else:
        op = reduced_ops[key]
      self.run_ops.append(op)
      self.name_ops.append(key)

    self.graph = tf.get_default_graph()
    self.global_step_op = self.graph.get_tensor_by_name("global_step:0")
    self.max_step_op = self.graph.get_tensor_by_name("max_step:0")

    self.saver = tf.train.Saver(
      max_to_keep=self.args.keep_checkpoint_max,
      name="global_saver")

  def before_run(self, callbacks, saver):
    for callback in callbacks:
      callback.before_run(self.sess, saver)

    self.run_feed_dict()

  def before_step(self, callbacks):
    for callback in callbacks:
      callback.before_step(self.sess)

  def after_step(self, callbacks, outputs_dict, saver):
    for callback in callbacks:
      callback.after_step(self.sess, outputs_dict, saver)

  def after_run(self, callbacks, saver):
    for callback in callbacks:
      callback.after_run(self.sess, saver)

  def run_feed_dict(self):
      for key in self.modeler.feed_dict_ops:
        self.feed_dict[key] = self.sess.run(
          self.modeler.feed_dict_ops[key])

  def run(self):
    self.create_graph()

    with tf.Session(config=self.session_config) as self.sess:

      # Before run
      self.before_run(self.modeler.callbacks, self.saver)

      global_step = self.sess.run(self.global_step_op)
      max_step = self.sess.run(self.max_step_op)

      while global_step < max_step:
        self.before_step(self.modeler.callbacks)

        outputs = self.sess.run(self.run_ops, feed_dict=self.feed_dict)
        global_step = self.sess.run(self.global_step_op)

        outputs_dict = {}
        for key, value in zip(self.name_ops, outputs):
          outputs_dict[key] = value

        self.after_step(self.modeler.callbacks, outputs_dict, self.saver)

      self.after_run(self.modeler.callbacks, self.saver)


def build(args, inputter, modeler):
  return ParameterServerRunner(args, inputter, modeler)
