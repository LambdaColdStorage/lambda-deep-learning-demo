Fully Convolutional Networks
========================================

* :ref:`downloaddata`
* :ref:`fcntrain`
* :ref:`fcneval`
* :ref:`fcninference`
* :ref:`fcntune`
* :ref:`fcnevalpretrain`
* :ref:`fcnexport`

.. _downloaddata:

Download CamVid Dataset
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz \
  --data_dir=~/demo/data

Fully Convolutional Networks
----------------------------------------------

.. _fcntrain:

Train from scratch
----------------------------------------------

::

  python demo/image/image_segmentation.py \
  --mode=train \
  --model_dir=~/demo/model/fcn_camvid \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --batch_size_per_gpu=16 --epochs=200 \
  train_args \
  --learning_rate=0.00129 --optimizer=adam \
  --piecewise_boundaries=100 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/camvid/train.csv


.. _fcneval:

Evaluation
----------------------------------------------

::

  python demo/image/image_segmentation.py \
  --mode=eval \
  --model_dir=~/demo/model/fcn_camvid \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --batch_size_per_gpu=4 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/camvid/val.csv



.. _fcninference:

Inference
----------------------------------------------

::

  python demo/image/image_segmentation.py \
  --mode=infer \
  --model_dir=~/demo/model/fcn_camvid \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  infer_args \
  --callbacks=infer_basic,infer_display_image_segmentation \
  --test_samples=~/demo/data/camvid/test/0001TP_008550.png,~/demo/data/camvid/test/Seq05VD_f02760.png,~/demo/data/camvid/test/Seq05VD_f04650.png,~/demo/data/camvid/test/Seq05VD_f05100.png


.. _fcntune:

Hyper-Parameter Tuning
----------------------------------------------

::

  python demo/image/image_segmentation.py \
  --mode=tune \
  --model_dir=~/demo/model/fcn_camvid \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --batch_size_per_gpu=16 \
  tune_args \
  --train_dataset_meta=~/demo/data/camvid/train.csv \
  --eval_dataset_meta=~/demo/data/camvid/val.csv \
  --tune_config=source/tool/fcn_camvid_tune_coarse.yaml



.. _fcnevalpretrain:

Evaluate Pre-trained model
----------------------------------------------

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/fcn_camvid_20190125.tar.gz | tar xvz -C ~/demo/model

::

  python demo/image/image_segmentation.py \
  --mode=eval \
  --model_dir=~/demo/model/fcn_camvid_20190125 \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --gpu_count=1 --batch_size_per_gpu=4 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/camvid/val.csv


.. _fcnexport:

Export
----------------------------------------------

::

  python demo/image/image_segmentation.py \
  --mode=export \
  --model_dir=~/demo/model/fcn_camvid_20190125 \
  --network=fcn \
  --augmenter=fcn_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  export_args \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_image \
  --output_ops=output_classes


.. _serve:

Serve
-------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_segmentation \
  --mount type=bind,source=model_dir/export,target=/models/segmenation \
  -e MODEL_NAME=segmentation -t tensorflow/serving:latest-gpu &

  python client/image_segmenation_client.py --image_path=path_to_image