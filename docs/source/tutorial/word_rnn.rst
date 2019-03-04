Word RNN
========================================


* :ref:`wordrnn_train`
* :ref:`wordrnn_eval`
* :ref:`wordrnn_inference`
* :ref:`wordrnn_tune`
* :ref:`wordrnn_export`
* :ref:`wordrnn_serve`


.. _wordrnn_downloaddata:

**Download Dataset**
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --data_dir=~/demo/data/

.. _wordrnn_buildvoc:

**Build Vocabulary**
----------------------------------------------

::

  python demo/text/preprocess/build_vocab_basic.py \
  --input_file=~/demo/data/shakespeare/shakespeare_input.txt \
  --output_vocab=~/demo/data/shakespeare/shakespeare_word_basic.vocab \
  --unit=word \
  --loader=word_basic

.. _wordrnn_train:

Train from scratch
----------------------------------------------

::

  python demo/text_generation.py \
  --mode=train \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=100 \
  --vocab_top_k=4000 \
  --unit=word \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=50 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt

.. _wordrnn_eval:

Evaluation
----------------------------------------------

::

  python demo/text_generation.py \
  --mode=eval \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=1 \
  --vocab_top_k=4000 \
  --unit=word \
  eval_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt

.. _wordrnn_infer:

Infer
----------------------------------------------

::

  python demo/text_generation.py \
  --mode=infer \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --unit=word \
  --vocab_top_k=4000 \
  infer_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --callbacks=infer_basic,infer_display_text_generation

.. _wordrnn_tune:

Hyper-Parameter Tuning
----------------------------------------------

::

  python demo/text_generation.py \
  --mode=tune \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --network=rnn_basic \
  --batch_size_per_gpu=128 \
  --unit=word \
  --vocab_top_k=4000 \
  tune_args \
  --train_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --eval_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --tune_config=source/tool/rnn_basic_shakespeare_tune_coarse.yaml

.. _wordrnn_export:

Export
----------------------------------------------

::

  python demo/text_generation.py \
  --mode=export \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --unit=word \
  --vocab_top_k=4000 \
  export_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_item,c0,h0,c1,h1 \
  --output_ops=output_probabilities,output_last_state,items