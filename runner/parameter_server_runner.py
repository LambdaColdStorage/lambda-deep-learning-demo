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

  def replicate_graph(self):
    with tf.device("/cpu:0"):
      self.modeler.create_precomputation()

      if self.args.mode == "infer":
        pass
      else:
        batch = self.inputter.input_fn()
        output = ()
        for i in range(self.args.num_gpu):
          with tf.device(self.assign_to_device("/gpu:{}".format(i),
                         ps_device="/cpu:0")):
            # Split input data across multiple devices
            x = self.batch_split(batch, i)

            y = self.modeler.model_fn(x)

            # Gather output across multiple devices
            if i == 0:
              for item_y in y:
                output = (output + ([item_y],))
            else:
              for item_y, item_output in zip(y, output):
                item_output.append(item_y)

        # Reduce
        reduced_ops = ()
        for x in output:
          reduced_ops = (reduced_ops + (self.reduce_op(x),))
        return reduced_ops

  def run(self):
    print("ParameterServerRunner is running.")
    reduced_ops = self.replicate_graph()

    run_ops = ()
    for op in reduced_ops:
      if isinstance(op, list):
        # Create train_op to minize the loss
        minimize_op = self.modeler.optimizer.apply_gradients(
          op, global_step=self.modeler.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        op = tf.group(minimize_op, update_ops)
      run_ops = (run_ops + (op,))

    with tf.Session(config=self.session_config) as self.sess:
      self.sess.run(tf.global_variables_initializer())
      for i in range(10):
        self.sess.run(run_ops)
        print(i)


def build(args, inputter, modeler):
  return ParameterServerRunner(args, inputter, modeler)
