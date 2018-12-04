"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

"""
import tensorflow as tf

from runner import Runner


class ParameterServerRunner(Runner):
  def __init__(self, config, inputter, modeler, callbacks):
    super(ParameterServerRunner, self).__init__(config,
                                                inputter,
                                                modeler,
                                                callbacks)
    self.ps_ops = ["Variable", "VariableV2", "AutoReloadVariable"]

  def assign_to_device(self, device, ps_device="/cpu:0"):
      def _assign(op):
          node_def = op if isinstance(op, tf.NodeDef) else op.node_def
          if node_def.op in self.ps_ops:
              return "/" + ps_device
          else:
              return device
      return _assign

  def batch_split(self, batch, idx):
    bs_per_gpu = self.config.batch_size_per_gpu
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

  def replicate_graph(self):

    batch = self.inputter.input_fn()

    if self.config.mode == "infer":
      with tf.device(self.assign_to_device("/gpu:{}".format(0),
                     ps_device="/cpu:0")):
        ops = self.modeler.model_fn(batch)
        return ops

    else:
      ops = {}
      if self.config.reduce_ops:
        output = {}
        # Map
        for i in range(self.config.gpu_count):
          with tf.device(self.assign_to_device("/gpu:{}".format(i),
                         ps_device="/cpu:0")):
            # Split input data across multiple devices
            x = self.batch_split(batch, i)
            y = self.modeler.model_fn(x)

            # Gather output across multiple devices
            if i == 0:
              for key in y:
                output[key] = [y[key]]
            else:
              for key in y:
                output[key].append(y[key])
        # Reduce
        for key in output:
          ops[key] = self.reduce_op(output[key])
      else:
        # Sometime the results from multiple GPUs are not to be reduced.
        # For example, in the case of saving results for evaluated by external tools.
        # Map
        for i in range(self.config.gpu_count):
          with tf.device(self.assign_to_device("/gpu:{}".format(i),
                         ps_device="/cpu:0")):
            # Split input data across multiple devices
            x = self.batch_split(batch, i)
            y = self.modeler.model_fn(x)
            # Gather output across multiple devices
            if i == 0:
              for key in y:
                ops[key] = y[key]
            else:
              for key in y:
                ops[key].extend(y[key])
      return ops

  def create_graph(self):

    with tf.device("/cpu:0"):

      nonreplicated_fns = [self.modeler.create_nonreplicated_fn,
                           self.inputter.create_nonreplicated_fn]

      for fn in nonreplicated_fns:
        fn()

      reduced_ops = self.replicate_graph()

      self.run_ops, self.run_ops_names = self.collect_ops(reduced_ops)

      self.graph = tf.get_default_graph()
      self.global_step_op = self.graph.get_tensor_by_name("global_step:0")
      self.max_step_op = self.graph.get_tensor_by_name("max_step:0")


def build(config, inputter, modeler, callbacks):
  return ParameterServerRunner(config, inputter, modeler, callbacks)
