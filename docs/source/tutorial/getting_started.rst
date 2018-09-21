Getting Started
========================================

Welcome to Lambda Lab's deep learning demo suite -- the place to find ready-to-use machine learnig models. We offer the following cool features:

* A curate of open-source, state-of-the-art models that cover major machine learning applications, including image classification, image segmentation, object detection etc.

* Pure Tensorflow implementation. Efforts are made to keep the boilplate consistent across all demos.

* Examples of transfer learning and how to adapt the model to new data.

In this getting started guide, we will walk you through the glossary of our code and the steps of building tensorflow applications.  

* :ref:`glossary`
* :ref:`example`

.. _glossary:

Glossary
--------------------------------------

Our TensorFlow application is comprised of three main components:

* **Inputter**: The data pipeline. It reads data from the disk, shuffles and preprocesses the data, creates batches, and does prefetching. An inputter is applicable to a specific problem that share the same type of input and output. For example, we have image_classification_inputter, machine_translation_inputter, object_detection_inputter ... etc. An inputter can optionally own an **augmenter** for data augmentation, for example, random scale and crop, color distortion ... etc.

* **Modeler**: The model pipeline. The Modeler encapsulates the forward pass and the computation of loss, gradient, and accuracy. Like the inputter, a modeler is applicable to a specific problem such as image classification or object detection. A modeler must own a **network** member that implements the network architecture, for example, an image classification modeler can have ResNet32, VGG19 or InceptionV4 as its network architecture.

* **Runner**: The job executor. It orchestrates the execution of an Inputter and a Modeler and distributes the workload across multiple hardware devices. It also uses **callbacks** to perform auxiliary tasks such as logging, model saving and result visualization.

:numref:`tensorflow-application` illustrates the composition of a tensorflow application using these building blocks.

.. figure:: images/tensorflow-application.png
   :name: tensorflow-application

.. _example:

Example: Training a ResNet32 newtowrk on CIFAR10 
---------------------------------------------------

Let's walk through an example of building a Tensorflow application. In this example we will use a ResNet32 model for classifying CIFAR10 images.


* :ref:`define_application`
* :ref:`define_inputter`
* :ref:`define_modeler`
* :ref:`define_runner`
* :ref:`run`

.. _define_application:

**Define an Application**
---------------------------------------------------

Before diving into details, here is all the code for this guide. It gives an overview of the boilplate we use to build Tensorflow applications. 

.. code-block:: python

  # Create basic inputter configration
  inputter_config = InputterConfig(
    mode="train",
    batch_size_per_gpu=64,
    gpu_count=1,    
    epochs=4,
    dataset_meta="~/demo/data/cifar10/train.csv \")

  # Add additional configuration for image classification
  inputter_config = ImageClassificationInputterConfig(
    inputter_config,
    image_height=32,
    image_width=32,
    image_depth=3,
    num_classes=10)


  # (Optionally) Create a augmenter.
  argmenter_name = "source.augmenter.cifar_augmenter"
  augmenter = importlib.import_module(argmenter_name)

  # Create a Inputter.
  inputter_name = "source.inputter.image_classification_csv_inputter"
  inputter = importlib.import_module(inputter_name).build(inputter_config, augmenter)


  # Create a ResNet32 network
  network_name = "source.network.resnet32"
  net = getattr(importlib.import_module(network_name), "net")

  # Create basic modeler configration
  modeler_config = ModelerConfig(
    mode="train",
    batch_size_per_gpu=64,
    gpu_count=1,    
    optimizer="momentum",
    learning_rate=0.01)

  # Add additional configuration for image classification
  modeler_config = ImageClassificationModelerConfig(
    modeler_config,
    num_classes=10)  

  # Create modeler
  modeler_name = "source.modeler.image_classification_modeler"
  modeler = importlib.import_module(modeler_name).build(modeler_config, net)

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

  # Run the application
  runner.run()

.. _define_inputter:

**Define an Inputter**
---------------------------------------------------

The inputter is the data pipeline. This example defines the data pipeline of feeding CIFAR10 data with some basic augmentations: 

.. code-block:: python

  # Create basic inputter configration
  inputter_config = InputterConfig(
    mode="train",
    batch_size_per_gpu=64,
    gpu_count=1,    
    epochs=4,
    dataset_meta="~/demo/data/cifar10/train.csv \")

  # Add additional configuration for image classification
  inputter_config = ImageClassificationInputterConfig(
    inputter_config,
    image_height=32,
    image_width=32,
    image_depth=3,
    num_classes=10)

  # (Optionally) Create a augmenter.
  argmenter_name = "source.augmenter.cifar_augmenter"
  augmenter = importlib.import_module(argmenter_name)

  # Create a Inputter.
  inputter_name = "source.inputter.image_classification_csv_inputter"
  inputter = importlib.import_module(inputter_name).build(inputter_config, augmenter)

* :code:`cifar_augmenter` random image cropping, flipping, brightness and contrast distortions. 
* :code:`inputter_config` sets arguments for the inputter. For example, whether it is used for training for evaluation, batch_size, the data path ... etc.
* :code:`inputter` is the data pipeline instance. It has an important :code:`input_fn` member function that produces a data generator.

The :code:`input_fn` of an image classification inputter looks like this:

.. code-block:: python

  def input_fn(self, test_samples=[]):

    # Get list of image paths and class labels
    samples = self.get_samples_fn()

    # Generate a Tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices(samples)
    
    # Shuffle the dataset for training
    if self.config.mode == "train":
      dataset = dataset.shuffle(self.config.shuffle_buffer_size)

    # Repeat the dataset for multiple epochs
    dataset = dataset.repeat(self.config.epochs)

    # Parse individal input sample, including reading image from path,
    # data augmentation
    dataset = dataset.map(
      lambda image, label: self.parse_fn(image, label),
      num_parallel_calls=4)

    # Batch data
    batch_size = (self.config.batch_size_per_gpu *
                  self.config.gpu_count)    
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    # Prefetch for efficiency
    dataset = dataset.prefetch(2)

    # Return data generator
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


.. _define_modeler:

**Define a Modeler**
---------------------------------------------------

The modeler defines the model pipeline. This example defines the computation that is needed for a ResNet32 network:

.. code-block:: python

  # Create a ResNet32 network
  network_name = "source.network.resnet32"
  net = getattr(importlib.import_module(network_name), "net")

  # Create basic modeler configration
  modeler_config = ModelerConfig(
    mode="train",
    batch_size_per_gpu=64,
    gpu_count=1,    
    optimizer="momentum",
    learning_rate=0.01)

  # Add additional configuration for image classification
  modeler_config = ImageClassificationModelerConfig(
    modeler_config,
    num_classes=10)  

  # Create modeler
  modeler_name = "source.modeler.image_classification_modeler"
  modeler = importlib.import_module(modeler_name).build(modeler_config, net)

* :code:`net` is the function that implments ResNet32's forward pass.
* :code:`modeler_config` contains the argments for building a ResNet32 model. Importantly, it sets up the number of classes.
* :code:`modeler` is the model pipeline. It has an important :code:`model_fn` member function that outputs a dictionary of operators to be run by a Tensorflow session.

The :code:`model_fn` for an image classification modeler looks like this:

.. code-block:: python

  def model_fn(self, x):

    # Input batch of images and labels
    images = x[0]
    labels = x[1]

    # Create graph for forward pass
    logits, predictions = self.create_graph_fn(images)

    # Return modeler operators
    if self.config.mode == "train":

      # Training mode returns operators for loss, gradient and accuracy
      loss = self.create_loss_fn(logits, labels)
      grads = self.create_grad_fn(loss)
      accuracy = self.create_eval_metrics_fn(
        predictions, labels)
      return {"loss": loss,
              "grads": grads,
              "accuracy": accuracy,
              "learning_rate": self.learning_rate}
    elif self.config.mode == "eval":

      # Evalution mode returns operators for loss and accuracy
      loss = self.create_loss_fn(logits, labels)
      accuracy = self.create_eval_metrics_fn(
        predictions, labels)
      return {"loss": loss,
              "accuracy": accuracy}
    elif self.config.mode == "infer":

      # Inference mode returns the predicted classes and probabilities for the predictions
      return {"classes": predictions["classes"],
              "probabilities": predictions["probabilities"]}


.. _define_runner:

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


.. _run:

**Run the Application**
---------------------------------------------------

To run the application, simply call :code:`runner.run()`. 

