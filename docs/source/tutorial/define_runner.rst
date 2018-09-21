**Define a Runner**
---------------------------------------------------

A runner runs the inputter and the modeler. It also use callbacks for auxiliary jobs:

.. code-block:: python

  # Create callback configuations
  callback_config = CallbackConfig(
    mode="train",
    batch_size_per_gpu=64,
    gpu_count=1,    
    model_dir="~/demo/model/image_classification_cifar10",
    log_every_n_iter=10,
    save_summary_steps=10)

  # Create callbacks
  callback_names = ["train_basic", "train_loss", "train_accuracy",
                    "train_speed", "train_summary"]
  callbacks = []
  for name in callback_names:
	callback = importlib.import_module(
	  "source.callback." + name).build(callback_config)
	callbacks.append(callback)

  # Create run config
  runner_config = RunnerConfig(
    mode="train",
    batch_size_per_gpu=64,
    gpu_count=1,    
    summary_names=["loss,accuracy", "learning_rate"])

  # Create a runner
  runner_name = "source.runner.parameter_server_runner"
  runner = importlib.import_module(runner_name).build(runner_config, inputter, modeler, callbacks)


There are two main tasks for a runner: First, running some operators in a Tensorflow session. Second, distributes the computation across multiple-devices if it is needed.

The :code:`run` member function implements the run:

.. code-block:: python

  def run(self):

    # Create the computation graph
    self.create_graph()

    # Create a Tensorflow session
    with tf.Session(config=self.session_config) as self.sess:

      # Do auxiliary jobs before running the graph
      self.before_run()

      # Set up the global step and the maximum step to run
      global_step = 0
      if self.config.mode == "train":
        # For resuming training from the last checkpoint
        global_step = self.sess.run(self.global_step_op)

      max_step = self.sess.run(self.max_step_op)

      # Run the job until max_step
      while global_step < max_step:

        # Do auxiliary jobs before running a step
        self.before_step()

        # Run a step
        self.outputs = self.sess.run(self.run_ops)

        # Do auxiliary jobs after running a step
        self.after_step()

        global_step = global_step + 1

      # Do auxiliary jobs after finishing the run
      self.after_run()

The second task is to distribute computation across multiple device if it is necessary. In this example we use dsynchronized multi-GPU training with a CPU as the parameter server. To do so we use a :code:`parameter_server_runner` that splits the input data across multiple-GPUs, run computation in parallel on these GPUs, and gather the results for parameter update. The key logic is implemented in its :code:`replicate_graph` member function.

.. code-block:: python

  def replicate_graph(self):

    # Fetch input daaa
    batch = self.inputter.input_fn()

    if self.config.mode == "infer":

      # Use a single GPU for inference 
      with tf.device(self.assign_to_device("/gpu:{}".format(0),
                     ps_device="/cpu:0")):
        ops = self.modeler.model_fn(batch)
        return ops

    else:

      output = {}
      # Distribute work across multiple GPUs
      for i in range(self.config.gpu_count):
        with tf.device(self.assign_to_device("/gpu:{}".format(i),
                       ps_device="/cpu:0")):

          # Get the split for the i-th GPU
          x = self.batch_split(batch, i)
          y = self.modeler.model_fn(x)

          # Gather output from the i-th GPU
          if i == 0:
            for key in y:
              output[key] = [y[key]]
          else:
            for key in y:
              output[key].append(y[key])

      # Average results
      reduced_ops = {}
      for key in output:
        reduced_ops[key] = self.reduce_op(output[key])

      # Return the operation to run averaged results
      return reduced_ops

    # Run the application
    runner.run()


To run the application, simply call :code:`runner.run()`. 

