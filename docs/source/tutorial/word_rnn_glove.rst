Word RNN with Glove Embedding
========================================

* :ref:`wordrnnglove_downloaddata`
* :ref:`wordrnnglove_downloadvocab`
* :ref:`wordrnnglove_train`
* :ref:`wordrnnglove_eval`
* :ref:`wordrnnglove_inference`
* :ref:`wordrnnglove_tune`
* :ref:`wordrnnglove_pretrain`
* :ref:`wordrnnglove_export`
* :ref:`wordrnnglove_serve`


.. _wordrnnglove_downloaddata:

Download Dataset
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
  --data_dir=~/demo/data/

.. _wordrnnglove_downloadvocab:

Download Glove Embedding
----------------------------------------------

::

  wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip -d ~/demo/model/glove.6B && rm glove.6B.zip

.. _wordrnnglove_train:

Train from scratch
----------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=train \
  --model_dir=~/demo/model/word_rnn_glove_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=10 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=5 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt

.. _wordrnnglove_eval:

Evaluation
----------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=eval \
  --model_dir=~/demo/model/word_rnn_glove_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=32 --epochs=1 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  eval_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt

.. _wordrnnglove_inference:

Inference
----------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=infer \
  --model_dir=~/demo/model/word_rnn_glove_shakespeare \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  --starter=king \
  --softmax_temperature=1.0 \
  infer_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --callbacks=infer_basic,infer_display_text_generation

.. _wordrnnglove_tune:

Hyper-Parameter Tuning
----------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=tune \
  --model_dir=~/demo/model/word_rnn_glove_shakespeare \
  --network=rnn_basic \
  --batch_size_per_gpu=128 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  tune_args \
  --train_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --eval_dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --tune_config=source/tool/rnn_basic_shakespeare_tune_coarse.yaml


.. _wordrnnglove_pretrain:

Inference Using Pre-trained model
---------------------------------------

Download pre-trained models:

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/word_rnn_glove_shakespeare-20190303.tar.gz | tar xvz -C ~/demo/model

Inference

::

  python demo/text/text_generation.py \
  --mode=infer \
  --model_dir=~/demo/model/word_rnn_glove_shakespeare-20190303 \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  --starter=king \
  --softmax_temperature=1.0 \
  infer_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --callbacks=infer_basic,infer_display_text_generation


.. _wordrnnglove_export:

Export
--------------------------------------------

::

  python demo/text/text_generation.py \
  --mode=export \
  --model_dir=~/demo/model/word_rnn_glove_shakespeare \
  --network=rnn_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  export_args \
  --dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_item,RNN/c0,RNN/h0,RNN/c1,RNN/h1 \
  --output_ops=output_logits,output_last_state


.. _wordrnnglove_serve:

Serve
------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_textgeneration \
  --mount type=bind,source=/home/ubuntu/demo/model/word_rnn_glove_shakespeare/export,target=/models/textgeneration \
  -e MODEL_NAME=textgeneration -t tensorflow/serving:latest-gpu &


  python client/text_generation_client.py \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_top_k=40000 \
  --vocab_format=txt \
  --unit=word --starter=the --length=256 --softmax_temperature=1.0