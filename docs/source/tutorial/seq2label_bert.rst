Sequence-to-label BERT
========================================


* :ref:`seq2label_bert_downloaddata`
* :ref:`seq2label_bert_preprocess`
* :ref:`seq2label_bert_downloadbert`
* :ref:`seq2label_bert_train`
* :ref:`seq2label_bert_eval`
* :ref:`seq2label_bert_inference`
* :ref:`seq2label_bert_tune`
* :ref:`seq2label_bert_pretrain`
* :ref:`seq2label_bert_export`
* :ref:`seq2label_bert_serve`


.. _seq2label_bert_downloaddata:

Download Dataset
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
  --data_dir=~/demo/data/

.. _seq2label_bert_preprocess:

Preprocess Dataset
----------------------------------------------

::

  python demo/text/preprocess/preprocess_aclImdb_v1.py \
  --remove_punctuation=False


.. _seq2label_bert_downloadbert:

Download Pre-trained BERT model
----------------------------------------------

::

  wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip && unzip uncased_L-12_H-768_A-12.zip -d ~/demo/model && rm uncased_L-12_H-768_A-12.zip


.. _seq2label_bert_train:

Train from scratch
----------------------------------------------

::

  python demo/text/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --network=seq2label_bert \
  --batch_size_per_gpu=16 --epochs=4 \
  --vocab_file=~/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_format=txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  --unit=word \
  --lr_method=linear_plus_warmup \
  train_args \
  --learning_rate=0.00002 --optimizer=custom \
  --piecewise_boundaries=1 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/IMDB/train.csv \
  --pretrained_model=~/demo/model/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --skip_pretrained_var=classification/output_weights,classification/output_bias,global_step,power,adam


.. _seq2label_bert_eval:

Evaluation
----------------------------------------------

::

  python demo/text/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --network=seq2label_bert \
  --batch_size_per_gpu=16 --epochs=1 \
  --vocab_file=~/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_format=txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  --unit=word \
  eval_args \
  --dataset_meta=~/demo/data/IMDB/test.csv

.. _seq2label_bert_inference:

Inference
---------------------

::

  python demo/text/text_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --network=seq2label_bert \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_format=txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  --unit=word \
  infer_args \
  --callbacks=infer_basic,infer_display_text_classification \
  --test_samples="This movie is awesome."#"This movie is bad."#"This movie has an unusual taste."#"It is not clear what this movie is about."#"This is not a very good movie."#"I saw this at the premier at TIFF and was thrilled to learn the story is about a real friendship." \
  --splitter=#


.. _seq2label_bert_tune:

Hyper-Parameter Tuning
---------------------------------

::

  python demo/text/text_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --network=seq2label_bert \
  --batch_size_per_gpu=16 \
  --vocab_file=~/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_format=txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  --unit=word \
  --lr_method=linear_plus_warmup \
  tune_args \
  --pretrained_model=~/demo/model/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --skip_pretrained_var=classification/output_weights,classification/output_bias,global_step,power,adam \
  --train_dataset_meta=~/demo/data/IMDB/train.csv \
  --eval_dataset_meta=~/demo/data/IMDB/test.csv \
  --tune_config=source/tool/seq2label_bert_IMDB_tune_coarse.yaml


.. _seq2label_bert_pretrain:

Evaluate Pre-trained model
---------------------------------------

Download pre-trained models:

::

  curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/seq2label_bert_Imdb-20190303.tar.gz | tar xvz -C ~/demo/model

Evaluate

::

  python demo/text/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_bert_Imdb-20190303 \
  --network=seq2label_bert \
  --batch_size_per_gpu=16 --epochs=1 \
  --vocab_file=~/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_format=txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  --unit=word \
  eval_args \
  --dataset_meta=~/demo/data/IMDB/test.csv


.. _seq2label_bert_export:

Export
---------------------------

::

  python demo/text/text_classification.py \
  --mode=export \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --network=seq2label_bert \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_format=txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  export_args \
  --dataset_meta=~/demo/data/IMDB/train.csv \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_text,input_mask \
  --output_ops=output_probabilities


.. _seq2label_bert_serve:

Serve
---------------------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_textclassification \
  --mount type=bind,source=/home/chuan/demo/model/seq2label_bert_Imdb/export,target=/models/textclassification \
  -e MODEL_NAME=textclassification -t tensorflow/serving:latest-gpu &

  python client/text_classification_client.py \
  --vocab_file=~/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_format=txt \
  --vocab_top_k=-1 \
  --encode_method=bert
