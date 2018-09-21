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

	.. toctree::
	   :maxdepth: 2

	   define_application
	   define_inputter
	   define_modeler
	   define_runner
