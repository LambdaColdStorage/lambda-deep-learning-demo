SSD
========================================

* :ref:`ssd_prepare`
* :ref:`ssd_downloadvgg`
* :ref:`ssd_train`
* :ref:`ssd_eval`
* :ref:`ssd_infer`
* :ref:`ssd_tune`
* :ref:`ssd_pretrain`
* :ref:`ssd_export`
* :ref:`ssd_serve`

.. _ssd_prepare:

Prepare
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


.. _ssd_downloadvgg:

Download VGG backbone
----------------------------------------------

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/VGG_16_reduce.tar.gz | tar xvz -C ~/demo/model


.. _ssd_train:

Train SSD from scratch on MSCOCO
----------------------------------------------

::

  python demo/image/object_detection.py \
  --mode=train --model_dir=~/demo/model/ssd300_mscoco \
  --network=ssd300 --augmenter=ssd_augmenter --batch_size_per_gpu=16 --epochs=100 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=300 \
  --feature_net=vgg_16_reduced --feature_net_path=demo/model/VGG_16_reduce/VGG_16_reduce.p \
  train_args --learning_rate=0.001 --optimizer=momentum --piecewise_boundaries=60,80 \
  --piecewise_lr_decay=1.0,0.1,0.01 --dataset_meta=train2014,valminusminival2014 \
  --callbacks=train_basic,train_loss,train_speed,train_summary \
  --skip_l2_loss_vars=l2_norm_scaler --summary_names=loss,learning_rate,class_losses,bboxes_losses

  python demo/image/object_detection.py \
  --mode=train --model_dir=~/demo/model/ssd512_mscoco \
  --network=ssd512 --augmenter=ssd_augmenter --batch_size_per_gpu=16 --epochs=100 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=512 \
  --feature_net=vgg_16_reduced --feature_net_path=demo/model/VGG_16_reduce/VGG_16_reduce.p \
  train_args --learning_rate=0.001 --optimizer=momentum --piecewise_boundaries=60,80 \
  --piecewise_lr_decay=1.0,0.1,0.01 --dataset_meta=train2014,valminusminival2014 \
  --callbacks=train_basic,train_loss,train_speed,train_summary \
  --skip_l2_loss_vars=l2_norm_scaler --summary_names=loss,learning_rate,class_losses,bboxes_losses

.. _ssd_eval:

Evaluate SSD on MSCOCO
--------------------------------

::

  python demo/image/object_detection.py \
  --mode=eval \
  --model_dir=~/demo/model/ssd300_mscoco \
  --network=ssd300 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=8 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco \
  --num_classes=81 --resolution=300 --confidence_threshold=0.01 \
  --feature_net=vgg_16_reduced \
  eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco

  python demo/image/object_detection.py \
  --mode=eval \
  --model_dir=~/demo/model/ssd512_mscoco \
  --network=ssd512 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=8 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco \
  --num_classes=81 --resolution=512 --confidence_threshold=0.01 \
  --feature_net=vgg_16_reduced \
  eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco

.. _ssd_infer:

Inference
-----------------------

::

  python demo/image/object_detection.py \
  --mode=infer --model_dir=~/demo/model/ssd300_mscoco \
  --network=ssd300 --augmenter=ssd_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=300 --confidence_threshold=0.5 \
  --feature_net=vgg_16_reduced \
  infer_args --callbacks=infer_basic,infer_display_object_detection \
  --test_samples=/mnt/data/data/mscoco/val2014/COCO_val2014_000000000042.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000073.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000074.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000133.jpg

  python demo/image/object_detection.py \
  --mode=infer --model_dir=~/demo/model/ssd512_mscoco \
  --network=ssd512 --augmenter=ssd_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco --num_classes=81 --resolution=512 --confidence_threshold=0.5 \
  --feature_net=vgg_16_reduced \
  infer_args --callbacks=infer_basic,infer_display_object_detection \
  --test_samples=/mnt/data/data/mscoco/val2014/COCO_val2014_000000000042.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000073.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000074.jpg,/mnt/data/data/mscoco/val2014/COCO_val2014_000000000133.jpg


.. _ssd_tune:

Hyper-Parameter Tuning
--------------------------------

::

  python demo/image/object_detection.py \
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


  python demo/image/object_detection.py \
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

.. _ssd_pretrain:

Evaluate Pre-trained model
---------------------------------------

Download pre-trained models:

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/ssd300_mscoco_20190105.tar.gz | tar xvz -C ~/demo/model

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/ssd512_mscoco_20190105.tar.gz | tar xvz -C ~/demo/model

Evaluate

::

  python demo/image/object_detection.py \
  --mode=eval \
  --model_dir=~/demo/model/ssd300_mscoco_20190105 \
  --network=ssd300 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=8 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco \
  --num_classes=81 --resolution=300 --confidence_threshold=0.01 \
  --feature_net=vgg_16_reduced \
  eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco

  python demo/image/object_detection.py \
  --mode=eval \
  --model_dir=~/demo/model/ssd512_mscoco_20190105 \
  --network=ssd512 \
  --augmenter=ssd_augmenter \
  --batch_size_per_gpu=8 --epochs=1 \
  --dataset_dir=/mnt/data/data/mscoco \
  --num_classes=81 --resolution=512 --confidence_threshold=0.01 \
  --feature_net=vgg_16_reduced \
  eval_args --dataset_meta=val2017 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco

.. _ssd_export:

Export
------------

::

  python demo/image/object_detection.py \
  --mode=export \
  --model_dir=~/demo/model/ssd300_mscoco_20190105 \
  --network=ssd300 \
  --augmenter=ssd_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --num_classes=81 --resolution=300 \
  --confidence_threshold 0.01 \
  --feature_net=vgg_16_reduced \
  export_args \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_image \
  --output_ops=output_scores,output_labels,output_bboxes

  python demo/image/object_detection.py \
  --mode=export \
  --model_dir=~/demo/model/ssd512_mscoco_20190105 \
  --network=ssd512 \
  --augmenter=ssd_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --num_classes=81 --resolution=512 \
  --confidence_threshold 0.01 \
  --feature_net=vgg_16_reduced \
  export_args \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_image \
  --output_ops=output_scores,output_labels,output_bboxes


.. _ssd_serve:

Serve
-------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_ \
  --mount type=bind,source=model_dir/export,target=/models/objectdetection \
  -e MODEL_NAME=objectdetection -t tensorflow/serving:latest-gpu &

  python client/image_segmenation_client.py --image_path=path_to_image  