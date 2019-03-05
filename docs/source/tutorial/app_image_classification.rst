Image Classification
========================================

* :ref:`imageclassification_downloaddata`
* :ref:`imageclassification_resnet32train`
* :ref:`imageclassification_resnet32eval`
* :ref:`imageclassification_resnet32infer`
* :ref:`imageclassification_resnet32tune`
* :ref:`imageclassification_resnet32pretrain`
* :ref:`imageclassification_export`
* :ref:`imageclassification_serve`

.. _imageclassification_downloaddata:

Download CIFAR10 Dataset
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz \
  --data_dir=~/demo/data

.. _imageclassification_resnet32train:

Train ResNet32 from scratch on CIFAR10
----------------------------------------------

::

  python demo/image/image_classification.py \
  --mode=train \
  --model_dir=~/demo/model/resnet32_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --batch_size_per_gpu=256 --epochs=100 \
  train_args \
  --learning_rate=0.5 --optimizer=momentum \
  --piecewise_boundaries=50,75,90 \
  --piecewise_lr_decay=1.0,0.1,0.01,0.001 \
  --dataset_meta=~/demo/data/cifar10/train.csv

.. _imageclassification_resnet32eval:

Evaluation
-----------------------

::

  python demo/image/image_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/resnet32_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --batch_size_per_gpu=128 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/cifar10/eval.csv

.. _imageclassification_resnet32infer:

Inference
-----------------------

::

  python demo/image/image_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/resnet32_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  infer_args \
  --callbacks=infer_basic,infer_display_image_classification \
  --test_samples=~/demo/data/cifar10/test/appaloosa_s_001975.png,~/demo/data/cifar10/test/domestic_cat_s_001598.png,~/demo/data/cifar10/test/rhea_s_000225.png,~/demo/data/cifar10/test/trucking_rig_s_001216.png

.. _imageclassification_resnet32tune:

Hyper-Parameter Tuning
---------------------------

::

  python demo/image/image_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/resnet32_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --batch_size_per_gpu=128 \
  tune_args \
  --train_dataset_meta=~/demo/data/cifar10/train.csv \
  --eval_dataset_meta=~/demo/data/cifar10/eval.csv \
  --tune_config=source/tool/resnet32_cifar10_tune_coarse.yaml

  python demo/image_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/resnet32_cifar10 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --batch_size_per_gpu=128 \
  tune_args \
  --train_dataset_meta=~/demo/data/cifar10/train.csv \
  --eval_dataset_meta=~/demo/data/cifar10/eval.csv \
  --tune_config=source/tool/resnet32_cifar10_tune_fine.yaml

.. _imageclassification_resnet32pretrain:

Evaluate Pre-trained model
------------------------------

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10-resnet32-20180824.tar.gz | tar xvz -C ~/demo/model

::

  python demo/image/image_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/cifar10-resnet32-20180824 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --batch_size_per_gpu=128 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/cifar10/eval.csv


.. _imageclassification_export:

Export
------------

::

  python demo/image/image_classification.py \
  --mode=export \
  --model_dir=~/demo/model/cifar10-resnet32-20180824 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  export_args \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_image \
  --output_ops=output_classes

.. _imageclassification_serve:

Serve
-------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_classification \
  --mount type=bind,source=model_dir/export,target=/models/classification \
  -e MODEL_NAME=classification -t tensorflow/serving:latest-gpu &

  python client/image_classification_client.py --image_path=path_to_image