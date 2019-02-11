Text Generation
========================================


* :ref:`char_rnn`

.. _char_rnn:


**Char RNN**
----------------------------------------------

Train from scratch

::

  python demo/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/word_rnn_CoLA \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=word_rnn \
  --batch_size_per_gpu=128 --epochs=100 \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=50 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/CoLA/in_domain_train.tsv


Evaluation

::


Infer

::


Hyper-Parameter Tuning

::



**Export**
------------

::