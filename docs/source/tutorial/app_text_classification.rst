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
  --model_dir=~/demo/model/seq2label_CoLA \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label \
  --batch_size_per_gpu=128 --epochs=100 \
  --vocab_file=/home/ubuntu/demo/data/CoLA/vocab.pkl \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=50 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/CoLA/in_domain_train.tsv


  python demo/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label \
  --batch_size_per_gpu=128 --epochs=100 \
  --vocab_file=/home/ubuntu/demo/data/aclImdb_v1/vocab.pkl \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=50 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=/home/ubuntu/demo/data/aclImdb_v1/train_clean.csv


  python demo/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_pretrain_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_pretrain \
  --batch_size_per_gpu=128 --epochs=100 \
  --vocab_file=/home/ubuntu/demo/data/aclImdb_v1/vocab.pkl \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --piecewise_boundaries=50 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=/home/ubuntu/demo/data/aclImdb_v1/train_clean.csv


  python demo/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_bert \
  --batch_size_per_gpu=16 --epochs=4 \
  train_args \
  --learning_rate=0.00002 --optimizer=custom \
  --piecewise_boundaries=1 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=/home/chuan/demo/data/IMDB/train.tf_record \
  --pretrained_model=/home/chuan/demo/model/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --skip_pretrained_var=classification/output_weights,classification/output_bias,global_step,power

/home/ubuntu/demo/model/glove.6B/glove.6B.50d.txt
/home/ubuntu/demo/model/uncased_L-12_H-768_A-12/vocab.txt
/home/ubuntu/demo/data/IMDB/vocab_basic.txt


# Working bert training
  python demo/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_bert \
  --batch_size_per_gpu=16 --epochs=4 \
  --vocab_file=/home/ubuntu/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  --lr_method=linear_plus_warmup \
  train_args \
  --learning_rate=0.00002 --optimizer=custom \
  --piecewise_boundaries=1 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=/home/ubuntu/demo/data/IMDB/train_clean.csv \
  --pretrained_model=/home/ubuntu/demo/model/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --skip_pretrained_var=classification/output_weights,classification/output_bias,global_step,power

# Working glove training ??
  python demo/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_glove_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 --epochs=100 \
  --vocab_file=/home/ubuntu/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --lr_method=linear_plus_warmup \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --dataset_meta=/home/ubuntu/demo/data/IMDB/train_clean.csv

# Working training from scrach ???
  python demo/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_basic_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 --epochs=100 \
  --vocab_file=/home/ubuntu/demo/data/IMDB/vocab_basic.txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --lr_method=linear_plus_warmup \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --dataset_meta=/home/ubuntu/demo/data/IMDB/train_clean.csv

Evaluation

::

  python demo/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_CoLA \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label \
  --batch_size_per_gpu=128 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/data/CoLA/vocab.pkl \
  eval_args \
  --dataset_meta=~/demo/data/CoLA/in_domain_dev.tsv


# Working basic evaluation ???
  python demo/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_basic_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/data/IMDB/vocab_basic.txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  eval_args \
  --dataset_meta=/home/ubuntu/demo/data/IMDB/test_clean.csv


  python demo/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_pretrain_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_pretrain \
  --batch_size_per_gpu=128 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/data/aclImdb_v1/vocab.pkl \
  eval_args \
  --dataset_meta=/home/ubuntu/demo/data/aclImdb_v1/test_clean.csv


# Working glove evaluation ???
  python demo/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_glove_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  eval_args \
  --dataset_meta=/home/ubuntu/demo/data/IMDB/test_clean.csv



  python demo/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_bert \
  --batch_size_per_gpu=16 --epochs=1 \
  eval_args \
  --dataset_meta=/home/chuan/demo/data/IMDB/eval.tf_record


# Working bert evaluation

  python demo/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_bert \
  --batch_size_per_gpu=16 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  eval_args \
  --dataset_meta=/home/ubuntu/demo/data/IMDB/test_clean.csv


Infer

::

  python demo/text_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/seq2label_CoLA \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/data/CoLA/vocab.pkl \
  infer_args \
  --callbacks=infer_basic,infer_display_text_classification \
  --test_samples='anson left before jenny saw himself .','they drank the pub .','the professor talked us .','the dog barked out of the room .','the more we study verbs , the crazier they get .','day by day the facts are getting murkier .'

Hyper-Parameter Tuning

::



**Export**
------------

::