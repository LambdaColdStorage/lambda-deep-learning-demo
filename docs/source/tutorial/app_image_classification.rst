Image Classification
========================================


.. **ResNet32 on CIFAR10**

.. * :ref:`resnet32train`
.. * :ref:`resnet32eval`

.. .. _resnet32train:

.. Train from scratch
.. --------------------------------------

.. ::

..   python demo/image_classification.py \
..   --mode=train \
..   --model_dir=~/demo/model/image_classification_cifar10 \
..   --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz \
..   --network=resnet32 \
..   --augmenter=cifar_augmenter \
..   --gpu_count=1 --batch_size_per_gpu=128 --epochs=100 \
..   train_args \
..   --learning_rate=0.5 --optimizer=momentum \
..   --piecewise_boundaries=75 \
..   --piecewise_lr_decay=1.0,0.1 \
..   --dataset_meta=~/demo/data/cifar10/train.csv


.. .. _resnet32eval:

.. Evaluation
.. ---------------------------------------------------


**ResNet32 on CIFAR10**

* :ref:`resnet32train`
* :ref:`resnet32eval`
* :ref:`resnet32infer`
* :ref:`resnet32tune`
* :ref:`resnet32pretrain`

.. _resnet32train:

Train from scratch
-----------------------

::

  python demo/image_classification.py \
  --mode=train \
  --model_dir=~/demo/model/image_classification_cifar10 \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=128 --epochs=100 \
  train_args \
  --learning_rate=0.5 --optimizer=momentum \
  --piecewise_boundaries=75 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/cifar10/train.csv

.. _resnet32eval:

Evaluation
-----------------------

::

  python demo/image_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/image_classification_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=128 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/cifar10/eval.csv

.. _resnet32infer:

Inference
-----------------------

::

  python demo/image_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/image_classification_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  infer_args \
  --test_samples=~/demo/data/cifar10/test/appaloosa_s_001975.png,~/demo/data/cifar10/test/domestic_cat_s_001598.png,~/demo/data/cifar10/test/rhea_s_000225.png,~/demo/data/cifar10/test/trucking_rig_s_001216.png

.. _resnet32tune:

Hyper-parameter tuning
-----------------------

::

  python demo/image_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/image_classification_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=128 \
  tune_args \
  --train_dataset_meta=~/demo/data/cifar10/train.csv \
  --eval_dataset_meta=~/demo/data/cifar10/eval.csv \
  --tune_config=source/tool/ResNet32_CIFAR10_tune_coarse.yaml

  python demo/image_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/image_classification_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=128 \
  tune_args \
  --train_dataset_meta=~/demo/data/cifar10/train.csv \
  --eval_dataset_meta=~/demo/data/cifar10/eval.csv \
  --tune_config=source/tool/ResNet32_CIFAR10_tune_fine.yaml

.. _resnet32pretrain:

Evaluate Pre-trained model
------------------------------

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10-resnet32-20180824.tar.gz | tar xvz -C ~/demo/model

  python demo/image_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/cifar10-resnet32-20180824 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=128 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/cifar10/eval.csv
