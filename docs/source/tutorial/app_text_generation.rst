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
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=100 \
  --vocab_top_k=-1 \
  --unit=char \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=50 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt


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


Evaluation

::

  python demo/text_generation.py \
  --mode=eval \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=1 \
  --vocab_top_k=-1 \
  --unit=char \
  eval_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt


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

Infer

::

  python demo/text_generation.py \
  --mode=infer \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --unit=char \
  --vocab_top_k=-1 \
  infer_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --callbacks=infer_basic,infer_display_text_generation


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
  

Hyper-Parameter Tuning

::

  python demo/text_generation.py \
  --mode=tune \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --network=rnn_basic \
  --batch_size_per_gpu=128 \
  --unit=char \
  --vocab_top_k=-1 \
  tune_args \
  --train_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --eval_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --tune_config=source/tool/rnn_basic_shakespeare_tune_coarse.yaml


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

**Export**
------------

::

  CUDA_VISIBLE_DEVICES=3 python demo/text_generation.py \
  --mode=export \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=char_rnn \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  export_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_chars,c0,h0,c1,h1 \
  --output_ops=output_probabilities,output_last_state,output_chars


  --output_ops=output_chars,output_probabilities,output_last_state