"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
from runner import Runner

import tensorflow as tf


class ParameterServerRunner(Runner):
  def __init__(self, args, inputter, modeler):
    super(ParameterServerRunner, self).__init__(args,
                                                inputter,
                                                modeler)
    self.ps_ops = ["Variable", "VariableV2", "AutoReloadVariable"]
    self.session_config = self.create_session_config()
    self.sess = None
    self.num_samples = inputter.get_num_samples()
    self.modeler.num_samples = self.num_samples

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

  def replicate_graph(self, pre_fn, input_fn, model_fn):
    with tf.device("/cpu:0"):
      pre_fn()

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

  def run(self):
    reduced_ops = self.replicate_graph(
      self.modeler.create_precomputation,
      self.inputter.input_fn,
      self.modeler.model_fn)

    # Create train_op for gradient, keep other ops unchanged
    run_ops = []
    name_ops = []
    for key in reduced_ops:
      if key == "grads":
        minimize_op = self.modeler.optimizer.apply_gradients(
          reduced_ops[key], global_step=self.modeler.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        op = tf.group(minimize_op, update_ops)
      else:
        op = reduced_ops[key]
      run_ops.append(op)
      name_ops.append(key)

    with tf.Session(config=self.session_config) as self.sess:
      self.sess.run(tf.global_variables_initializer())

      self.feed_dict_pre = {}
      for key in self.modeler.pre_compute_ops:
        self.feed_dict_pre[key] = self.sess.run(
          self.modeler.pre_compute_ops[key])
      for i in range(100):
        outputs = self.sess.run(run_ops, feed_dict=self.feed_dict_pre)
        print(outputs[0])


def build(args, inputter, modeler):
  return ParameterServerRunner(args, inputter, modeler)
