Style Transfer
========================================

* :ref:`fns`

.. _fns:


**Fast Neural Style**
----------------------------------------------

Train from scratch

::

  python demo/style_transfer.py \
  --mode=train \
  --model_dir=~/demo/model/fns_gothic \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/mscoco_fns.tar.gz \
  --network=fns \
  --augmenter=fns_augmenter \
  --batch_size_per_gpu=4 --epochs=10 \
  train_args \
  --learning_rate=0.01 --optimizer=rmsprop \
  --piecewise_boundaries=5 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/mscoco_fns/train2014.csv \
  --summary_names=loss,learning_rate \
  --callbacks=train_basic,train_loss,train_speed,train_summary \
  --trainable_vars=FNS


Evaluation

::

  python demo/style_transfer.py \
  --mode=eval \
  --model_dir=~/demo/model/fns_gothic \
  --network=fns \
  --augmenter=fns_augmenter \
  --batch_size_per_gpu=4 --epochs=1 \
  eval_args \
  --callbacks=eval_basic,eval_loss,eval_speed,eval_summary \
  --dataset_meta=~/demo/data/mscoco_fns/eval2014.csv
  

Inference

::

  python demo/style_transfer.py \
  --mode=infer \
  --model_dir=~/demo/model/fns_gothic \
  --network=fns \
  --augmenter=fns_augmenter \
  --batch_size_per_gpu=1 --epochs=1 --gpu_count=1 \
  infer_args \
  --callbacks=infer_basic,infer_display_style_transfer \
  --test_samples=~/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg


Hyper-Parameter Tuning

::

  python demo/style_transfer.py \
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
  --tune_config=source/tool/fns_gothic_tune_coarse.yaml
