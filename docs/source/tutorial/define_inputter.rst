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

* :code:`cifar_augmenter` does random image cropping, flipping, brightness and contrast distortions. 
* :code:`inputter_config` sets arguments for the inputter. For example, whether it is used for training or evaluation, batch_size, the data path ... etc.
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
      dataset = dataset.shuffle(self.get_num_samples())

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