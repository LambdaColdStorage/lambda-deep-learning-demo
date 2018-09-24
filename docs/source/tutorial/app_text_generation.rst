Text Generation
========================================


* :ref:`char_rnn`

.. _char_rnn:


**Char RNN**
----------------------------------------------

Train from scratch

::

  python demo/text_generation.py \
  --mode=train \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --network=char_rnn \
  --gpu_count=1 --batch_size_per_gpu=128 --epochs=20 \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=10 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt


Evaluation

::

  python demo/text_generation.py \
  --mode=eval \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=char_rnn \
  --gpu_count=1 --batch_size_per_gpu=128 --epochs=1 \
  eval_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt
  

Infer

::

  python demo/text_generation.py \
  --mode=infer \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=char_rnn \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  infer_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --callbacks=infer_basic,infer_display_text_generation
  

Hyper-Parameter Tuning

::

  python demo/text_generation.py \
  --mode=tune \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=char_rnn \
  --gpu_count=1 --batch_size_per_gpu=128 \
  tune_args \
  --train_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --eval_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --tune_config=source/tool/char_rnn_shakespeare_tune_coarse.yaml