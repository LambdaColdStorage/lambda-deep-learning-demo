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

  def assign_to_device(self, device, ps_device="/cpu:0"):
      def _assign(op):
          node_def = op if isinstance(op, tf.NodeDef) else op.node_def
          if node_def.op in self.ps_ops:
              return "/" + ps_device
          else:
              return device
      return _assign

  def replicate_graph(self):
    bs_per_gpu = self.args.batch_size_per_gpu

    with tf.device("/cpu:0"):
      self.modeler.create_precomputation()

      self.global_step = tf.train.get_or_create_global_step()

      if self.args.mode == "infer":
        pass
      else:
        self.batch = self.inputter.input_fn()
        tower_losses = []
        tower_grads = []
        tower_accuracies = []

        print(type(self.batch))

        # for i in range(self.args.num_gpu):
        #   with tf.device(self.assign_to_device("/gpu:{}".format(i),
        #                  ps_device="/cpu:0")):


  def run(self):
    print("ParameterServerRunner is running.")
    self.replicate_graph()

    # with tf.Session(config=self.session_config) as self.sess:
    #   for i in range(10):
    #     _batch = self.sess.run(self.batch)
    #     print(_batch)

def build(args, inputter, modeler):
  return ParameterServerRunner(args, inputter, modeler)
