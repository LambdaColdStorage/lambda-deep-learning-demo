Object Detection
========================================

* :ref:`prepare`
* :ref:`ssdtrain`
* :ref:`ssdeval`
* :ref:`ssdinfer`
* :ref:`ssdtune`
* :ref:`ssdpretrain`


.. _prepare:

**Prepare**
----------------------------------------------
Install cocoapi_. 

::

  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  make install


Download coco dataset_.

* Download train2014, val2014, val2017 data and annotations.
* Uncompress them into your local machine. We use "/mnt/data/data/mscoco" as the data path in the following examples.

.. _cocoapi: https://github.com/cocodataset/cocoapi
.. _dataset: http://cocodataset.org/#download

.. _ssdtrain:

**Train SSD from scratch on MSCOCO**
----------------------------------------------

::
  
  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/VGG_16_reduce.tar.gz | tar xvz -C ~/demo/model


::

  python demo/object_detection.py \
  --mode=train --model_dir=~/demo/model/ssd300_mscoco \
  --network=ssd300 --augmenter=ssd_augmenter --batch_size_per_gpu=16 --epochs=100 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=300 \
  --feature_net=vgg_16_reduced --feature_net_path=demo/model/VGG_16_reduce/VGG_16_reduce.p \
  train_args --learning_rate=0.001 --optimizer=momentum --piecewise_boundaries=60,80 \
  --piecewise_lr_decay=1.0,0.1,0.01 --dataset_meta=train2014,valminusminival2014 \
  --callbacks=train_basic,train_loss,train_speed,train_summary \
  --skip_l2_loss_vars=l2_norm_scaler --summary_names=loss,learning_rate,class_losses,bboxes_losses

  python demo/object_detection.py \
  --mode=train --model_dir=~/demo/model/ssd512_mscoco \
  --network=ssd512 --augmenter=ssd_augmenter --batch_size_per_gpu=16 --epochs=100 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=512 \
  --feature_net=vgg_16_reduced --feature_net_path=demo/model/VGG_16_reduce/VGG_16_reduce.p \
  train_args --learning_rate=0.001 --optimizer=momentum --piecewise_boundaries=60,80 \
  --piecewise_lr_decay=1.0,0.1,0.01 --dataset_meta=train2014,valminusminival2014 \
  --callbacks=train_basic,train_loss,train_speed,train_summary \
  --skip_l2_loss_vars=l2_norm_scaler --summary_names=loss,learning_rate,class_losses,bboxes_losses

.. _ssdeval:

**Evaluate SSD on MSCOCO**
--------------------------------

::

  python demo/object_detection.py \
  --mode=eval \
  --model_dir=~/demo/model/ssd300_mscoco \
  --network=ssd300 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=8 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco \
  --num_classes=81 --resolution=300 --confidence_threshold=0.01 \
  eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco

  python demo/object_detection.py \
  --mode=eval \
  --model_dir=~/demo/model/ssd512_mscoco \
  --network=ssd512 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=8 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco \
  --num_classes=81 --resolution=512 --confidence_threshold=0.01 \
  eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco

.. _ssdinfer:

**Inference**
-----------------------

::

  python demo/object_detection.py \
  --mode=infer --model_dir=~/demo/model/ssd300_mscoco \
  --network=ssd300 --augmenter=ssd_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=300 --confidence_threshold=0.5 \
  infer_args --callbacks=infer_basic,infer_display_object_detection \
  --test_samples=/mnt/data/data/mscoco/val2014/COCO_val2014_000000000042.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000073.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000074.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000133.jpg

  python demo/object_detection.py \
  --mode=infer --model_dir=~/demo/model/ssd512_mscoco \
  --network=ssd512 --augmenter=ssd_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=512 --confidence_threshold=0.5 \
  infer_args --callbacks=infer_basic,infer_display_object_detection \
  --test_samples=/mnt/data/data/mscoco/val2014/COCO_val2014_000000000042.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000073.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000074.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000133.jpg


.. _ssdtune:

**Hyper-Parameter Tuning**
--------------------------------

::

  python demo/object_detection.py \
  --mode=tune \
  --model_dir=~/demo/model/ssd300mscoco \
  --network=ssd300 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=16 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=300 \
  --feature_net=vgg_16_reduced --feature_net_path=demo/model/VGG_16_reduce/VGG_16_reduce.p \
  tune_args \
  --train_callbacks=train_basic,train_loss,train_speed,train_summary \
  --eval_callbacks=eval_basic,eval_speed,eval_mscoco \
  --train_dataset_meta=train2017 \
  --eval_dataset_meta=val2017 \
  --tune_config=source/tool/ssd300_mscoco_tune_coarse.yaml \
  --eval_reduce_ops=False \
  --trainable_vars=SSD \
  --skip_l2_loss_vars=l2_norm_scaler


  python demo/object_detection.py \
  --mode=tune \
  --model_dir=~/demo/model/ssd512_mscoco \
  --network=ssd512 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=16 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=512\
  --feature_net=vgg_16_reduced --feature_net_path=demo/model/VGG_16_reduce/VGG_16_reduce.p \
  tune_args \
  --train_callbacks=train_basic,train_loss,train_speed,train_summary \
  --eval_callbacks=eval_basic,eval_speed,eval_mscoco \
  --train_dataset_meta=train2017 \
  --eval_dataset_meta=val2017 \
  --tune_config=source/tool/ssd512_mscoco_tune_coarse.yaml \
  --eval_reduce_ops=False \
  --trainable_vars=SSD \
  --skip_l2_loss_vars=l2_norm_scaler

.. _ssdpretrain:

**Evaluate Pre-trained model**
---------------------------------------

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/ssd300_mscoco_20190105.tar.gz | tar xvz -C ~/demo/model

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/ssd512_mscoco_20190105.tar.gz | tar xvz -C ~/demo/model

::

  python demo/object_detection.py \
  --mode=eval \
  --model_dir=~/demo/model/ssd300_mscoco_20190105 \
  --network=ssd300 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=8 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco \
  --num_classes=81 --resolution=300 --confidence_threshold=0.01 \
  eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco

  python demo/object_detection.py \
  --mode=eval \
  --model_dir=~/demo/model/ssd512_mscoco_20190105 \
  --network=ssd512 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=8 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco \
  --num_classes=81 --resolution=512 --confidence_threshold=0.01 \
  eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco