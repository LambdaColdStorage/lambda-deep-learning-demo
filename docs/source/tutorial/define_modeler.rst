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
