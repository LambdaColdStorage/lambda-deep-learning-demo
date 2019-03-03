Fast Neural Style
========================================

* :ref:`downloadvgg`
* :ref:`downloadmscocosub`
* :ref:`fnstrain`
* :ref:`fnseval`
* :ref:`fnsinference`
* :ref:`fnstune`
* :ref:`fnsevalpretrain`
* :ref:`fnsexport`

.. _downloadvgg:

Download VGG backbone
----------------------------------------------

::

  (mkdir ~/demo/model/vgg_19_2016_08_28;curl http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz | tar xvz -C ~/demo/model/vgg_19_2016_08_28)


.. _downloadmscocosub:

Download MSCOCO (sub) Dataset
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/mscoco_fns.tar.gz \
  --data_dir=~/demo/data


.. _fnstrain:

Train from scratch
----------------------------------------------

::

  python demo/image/style_transfer.py \
  --mode=train \
  --model_dir=~/demo/model/fns_gothic \
  --network=fns \
  --augmenter=fns_augmenter \
  --batch_size_per_gpu=8 --epochs=100 \
  train_args \
  --learning_rate=0.00185 --optimizer=rmsprop \
  --piecewise_boundaries=90 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/mscoco_fns/train2014.csv \
  --summary_names=loss,learning_rate \
  --callbacks=train_basic,train_loss,train_speed,train_summary \
  --trainable_vars=FNS

.. _fnseval:

Evaluation
----------------------------------------------

::

  python demo/image/style_transfer.py \
  --mode=eval \
  --model_dir=~/demo/model/fns_gothic \
  --network=fns \
  --augmenter=fns_augmenter \
  --batch_size_per_gpu=4 --epochs=1 \
  eval_args \
  --callbacks=eval_basic,eval_loss,eval_speed,eval_summary \
  --dataset_meta=~/demo/data/mscoco_fns/eval2014.csv
  

.. _fnsinference:

Inference
----------------------------------------------

::

  python demo/image/style_transfer.py \
  --mode=infer \
  --model_dir=~/demo/model/fns_gothic \
  --network=fns \
  --augmenter=fns_augmenter \
  --batch_size_per_gpu=1 --epochs=1 --gpu_count=1 \
  infer_args \
  --callbacks=infer_basic,infer_display_style_transfer \
  --test_samples=~/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000015070.jpg


.. _fnstune:

Hyper-Parameter Tuning
----------------------------------------------

::

  python demo/image/style_transfer.py \
  --mode=tune \
  --model_dir=~/demo/model/fns_gothic \
  --network=fns \
  --augmenter=fns_augmenter \
  --batch_size_per_gpu=4 \
  tune_args \
  --train_dataset_meta=~/demo/data/mscoco_fns/train2014.csv \
  --eval_dataset_meta=~/demo/data/mscoco_fns/eval2014.csv \
  --train_callbacks=train_basic,train_loss,train_speed,train_summary \
  --eval_callbacks=eval_basic,eval_loss,eval_speed,eval_summary \
  --tune_config=source/tool/fns_gothic_tune_coarse.yaml \
  --trainable_vars=FNS


.. _fnsevalpretrain:

Evaluate Pre-trained model
----------------------------------------------

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/fns_gothic_20190126.tar.gz | tar xvz -C ~/demo/model


  python demo/image/style_transfer.py \
  --mode=infer \
  --model_dir=~/demo/model/fns_gothic_20190126 \
  --network=fns \
  --augmenter=fns_augmenter \
  --batch_size_per_gpu=1 --epochs=1 --gpu_count=1 \
  infer_args \
  --callbacks=infer_basic,infer_display_style_transfer \
  --test_samples=~/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000015070.jpg


.. _fnsexport:

Export
----------------------------------------------

::
  python demo/image/style_transfer.py \
  --mode=export \
  --model_dir=~/demo/model/fns_gothic_20190126 \
  --network=fns \
  --augmenter=fns_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  export_args \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_image \
  --output_ops=output_image


.. _serve:

Serve
-------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_styletransfer \
  --mount type=bind,source=model_dir/export,target=/models/styletransfer \
  -e MODEL_NAME=styletransfer -t tensorflow/serving:latest-gpu &

  python client/style_transfer_client.py --image_path=path_to_image