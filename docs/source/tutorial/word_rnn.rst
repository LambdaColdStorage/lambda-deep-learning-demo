Word RNN
========================================

* :ref:`wordrnn_downloaddata`
* :ref:`wordrnn_buildvoc`
* :ref:`wordrnn_train`
* :ref:`wordrnn_eval`
* :ref:`wordrnn_inference`
* :ref:`wordrnn_tune`
* :ref:`wordrnn_pretrain`
* :ref:`wordrnn_export`
* :ref:`wordrnn_serve`


.. _wordrnn_downloaddata:

Download Dataset
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --data_dir=~/demo/data/

.. _wordrnn_buildvoc:

Build Vocabulary
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

  python demo/text/text_generation.py \
  --mode=train \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=128 --epochs=10 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --encode_method=basic \
  --unit=word \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=5 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt

.. _wordrnn_eval:

Evaluation
----------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=eval \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=1 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --encode_method=basic \
  --unit=word \
  eval_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt

.. _wordrnn_inference:

Inference
----------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=infer \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --unit=word \
  --starter=The \
  --softmax_temperature=1.0 \
  infer_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --callbacks=infer_basic,infer_display_text_generation

.. _wordrnn_tune:

Hyper-Parameter Tuning
----------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=tune \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=128 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --unit=word \
  tune_args \
  --train_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --eval_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --tune_config=source/tool/rnn_basic_shakespeare_tune_coarse.yaml


.. _wordrnn_pretrain:

Inference Using Pre-trained model
---------------------------------------

Download pre-trained models:

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/word_rnn_shakespeare-20190303.tar.gz | tar xvz -C ~/demo/model

Inference

::

  python demo/text/text_generation.py \
  --mode=infer \
  --model_dir=~/demo/model/word_rnn_shakespeare-20190303 \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --unit=word \
  --starter=The \
  --softmax_temperature=1.0 \
  infer_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --callbacks=infer_basic,infer_display_text_generation

.. _wordrnn_export:

Export
----------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=export \
  --model_dir=~/demo/model/word_rnn_shakespeare \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/data/shakespeare/shakespeare_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=-1 \
  --unit=word \
  export_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_item,RNN/c0,RNN/h0,RNN/c1,RNN/h1 \
  --output_ops=output_logits,output_last_state

.. _wordrnn_serve:

Serve
------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_textgeneration \
  --mount type=bind,source=/home/ubuntu/demo/model/word_rnn_shakespeare/export,target=/models/textgeneration \
  -e MODEL_NAME=textgeneration -t tensorflow/serving:latest-gpu &


  python client/text_generation_client.py \
  --vocab_file=~/demo/data/shakespeare/shakespeare_word_basic.vocab \
  --vocab_top_k=-1 \
  --vocab_format=pickle \
  --unit=word --starter=KING --length=256 --softmax_temperature=1.0