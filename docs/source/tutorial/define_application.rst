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