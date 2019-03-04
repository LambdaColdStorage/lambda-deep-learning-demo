Char RNN
========================================


* :ref:`charrnn_train`
* :ref:`charrnn_eval`
* :ref:`charrnn_inference`
* :ref:`charrnn_tune`
* :ref:`charrnn_export`
* :ref:`charrnn_serve`


.. _charrnn_downloaddata:

**Download Dataset**
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --data_dir=~/demo/data/

.. _charrnn_buildvoc:

**Build Vocabulary**
----------------------------------------------

::

  python demo/text/preprocess/build_vocab_basic.py \
  --input_file=~/demo/data/shakespeare/shakespeare_input.txt \
  --output_vocab=~/demo/data/shakespeare/shakespeare_char_basic.vocab \
  --unit=char \
  --loader=char_basic

.. _charrnn_train:

Train from scratch
-------------------------------

::

  python demo/text/text_generation.py \
  --mode=train \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=100 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_char_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --encode_method=basic \
  --unit=char \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=50 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt

.. _charrnn_eval:

Evaluation
-------------------------------

::

  python demo/text/text_generation.py \
  --mode=eval \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=1 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_char_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --encode_method=basic \
  --unit=char \
  eval_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt

.. _charrnn_inference:

Inference
-------------------------------

::

  python demo/text/text_generation.py \
  --mode=infer \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_char_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --unit=char \
  --starter=V \
  --softmax_temperature=1.0 \
  infer_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --callbacks=infer_basic,infer_display_text_generation

.. _charrnn_tune:

Hyper-Parameter Tuning
-------------------------------

::

  python demo/text/text_generation.py \
  --mode=tune \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=128 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_char_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --unit=char \
  tune_args \
  --train_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --eval_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --tune_config=source/tool/rnn_basic_shakespeare_tune_coarse.yaml

.. _charrnn_export:

Export
------------

::

  python demo/text/text_generation.py \
  --mode=export \
  --model_dir=~/demo/model/char_rnn_shakespeare \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_char_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --unit=char \
  export_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_item,RNN/c0,RNN/h0,RNN/c1,RNN/h1 \
  --output_ops=output_logits,output_last_state


Serve
------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_textgeneration \
  --mount type=bind,source=/home/ubuntu/demo/model/char_rnn_shakespeare/export,target=/models/textgeneration \
  -e MODEL_NAME=textgeneration -t tensorflow/serving:latest-gpu &


  python client/text_generation_client.py \
  --vocab_file=~/demo/data/shakespeare/shakespeare_char_basic.vocab \
  --vocab_top_k=-1 \
  --vocab_format=pickle \
  --unit=char --starter=V --length=1000 --softmax_temperature=1.0