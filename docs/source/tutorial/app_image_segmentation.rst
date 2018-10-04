Image Segmenation
========================================


* :ref:`fcn`
* :ref:`unet`

.. _fcn:


**Fully Convolutional Networks**
----------------------------------------------

Train from scratch

::

  python demo/image_segmentation.py \
  --mode=train \
  --model_dir=~/demo/model/fcn_camvid \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --batch_size_per_gpu=16 --epochs=200 --gpu_count=1 \
  train_args \
  --learning_rate=0.00129 --optimizer=adam \
  --piecewise_boundaries=100 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/camvid/train.csv

Evaluation

::

  python demo/image_segmentation.py \
  --mode=eval \
  --model_dir=~/demo/model/fcn_camvid \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --batch_size_per_gpu=4 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/camvid/val.csv


Inference

::

  python demo/image_segmentation.py \
  --mode=infer \
  --model_dir=~/demo/model/fcn_camvid \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  infer_args \
  --callbacks=infer_basic,infer_display_image_segmentation \
  --test_samples=~/demo/data/camvid/test/0001TP_008550.png,~/demo/data/camvid/test/Seq05VD_f02760.png,~/demo/data/camvid/test/Seq05VD_f04650.png,~/demo/data/camvid/test/Seq05VD_f05100.png

Hyper-Parameter Tuning

::

  python demo/image_segmentation.py \
  --mode=tune \
  --model_dir=~/demo/model/fcn_camvid \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --batch_size_per_gpu=16  --gpu_count=1 \
  tune_args \
  --train_dataset_meta=~/demo/data/camvid/train.csv \
  --eval_dataset_meta=~/demo/data/camvid/val.csv \
  --tune_config=source/tool/fcn_camvid_tune_coarse.yaml

::

  python demo/image_segmentation.py \
  --mode=tune \
  --model_dir=~/demo/model/fcn_camvid \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --batch_size_per_gpu=16  --gpu_count=1 \
  tune_args \
  --train_dataset_meta=~/demo/data/camvid/train.csv \
  --eval_dataset_meta=~/demo/data/camvid/val.csv \
  --tune_config=source/tool/fcn_camvid_tune_fine.yaml

.. _unet:

**U-Net**
----------------------------------------------

Train from scratch

::

  python demo/image_segmentation.py \
  --mode=train \
  --model_dir=~/demo/model/unet_camvid \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz \
  --network=unet \
  --augmenter=unet_augmenter \
  --batch_size_per_gpu=16 --epochs=200 --gpu_count=1 \
  train_args \
  --learning_rate=0.00129 --optimizer=adam \
  --piecewise_boundaries=100 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/camvid/train.csv

Evaluation

::

  python demo/image_segmentation.py \
  --mode=eval \
  --model_dir=~/demo/model/unet_camvid \
  --network=unet \
  --augmenter=unet_augmenter \
  --batch_size_per_gpu=4 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/camvid/val.csv


Inference

::

  python demo/image_segmentation.py \
  --mode=infer \
  --model_dir=~/demo/model/unet_camvid \
  --network=unet \
  --augmenter=unet_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  infer_args \
  --callbacks=infer_basic,infer_display_image_segmentation \
  --test_samples=~/demo/data/camvid/test/0001TP_008550.png,~/demo/data/camvid/test/Seq05VD_f02760.png,~/demo/data/camvid/test/Seq05VD_f04650.png,~/demo/data/camvid/test/Seq05VD_f05100.png


Hyper-Parameter Tuning

::

  python demo/image_segmentation.py \
  --mode=tune \
  --model_dir=~/demo/model/unet_camvid \
  --network=unet \
  --augmenter=unet_augmenter \
  --gpu_count=1 --batch_size_per_gpu=16 \
  tune_args \
  --train_dataset_meta=~/demo/data/camvid/train.csv \
  --eval_dataset_meta=~/demo/data/camvid/val.csv \
  --tune_config=source/tool/unet_camvid_tune_coarse.yaml

::

  python demo/image_segmentation.py \
  --mode=tune \
  --model_dir=~/demo/model/unet_camvid \
  --network=unet \
  --augmenter=unet_augmenter \
  --gpu_count=1 --batch_size_per_gpu=16 \
  tune_args \
  --train_dataset_meta=~/demo/data/camvid/train.csv \
  --eval_dataset_meta=~/demo/data/camvid/val.csv \
  --tune_config=source/tool/unet_camvid_tune_fine.yaml