Sequence-to-label Basic
========================================


* :ref:`seq2label_basic_downloaddata`
* :ref:`seq2label_basic_preprocess`
* :ref:`seq2label_basic_buildvoc`
* :ref:`seq2label_basic_train`
* :ref:`seq2label_basic_eval`
* :ref:`seq2label_basic_inference`
* :ref:`seq2label_basic_tune`
* :ref:`seq2label_basic_export`
* :ref:`seq2label_basic_serve`


.. _seq2label_basic_downloaddata:

Download Dataset
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
  --data_dir=~/demo/data/


.. _seq2label_basic_preprocess:

Preprocess Dataset
---------------------------------------------

::

  python demo/text/preprocess/preprocess_aclImdb_v1.py \
  --remove_punctuation=False


.. _seq2label_basic_buildvoc:

Build Vocabulary
----------------------------------------------

::

  python demo/text/preprocess/build_vocab_aclImdb_v1.py \
  --input_file=~/demo/data/IMDB/train.csv \
  --output_vocab=~/demo/data/IMDB/imdb_word_basic.vocab \
  --unit=word \
  --loader=imdb_loader

.. _seq2label_basic_train:

Train from scratch
----------------------------------------------

::

  python demo/text/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_basic_Imdb \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 --epochs=100 \
  --vocab_file=~/demo/data/IMDB/imdb_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  --lr_method=linear_plus_warmup \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --dataset_meta=~/demo/data/IMDB/train.csv


.. _seq2label_basic_eval:

Evaluation
----------------------------------------------

::

  python demo/text/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_basic_Imdb \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 --epochs=1 \
  --vocab_file=~/demo/data/IMDB/imdb_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  eval_args \
  --dataset_meta=~/demo/data/IMDB/test.csv


.. _seq2label_basic_inference:

Inference
---------------------

::

  python demo/text/text_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/seq2label_basic_Imdb \
  --network=seq2label_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/data/IMDB/imdb_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=40000 \
  --encode_method=basic \
  infer_args \
  --callbacks=infer_basic,infer_display_text_classification \
  --test_samples="This movie is awesome."#"This movie is bad."#"This movie has an unusual taste."#"It is not clear what this movie is about."#"This is not a very good movie."#"I saw this at the premier at TIFF and was thrilled to learn the story is about a real friendship." \
  --splitter=#


.. _seq2label_basic_tune:

Hyper-Parameter Tuning
---------------------------------

::

  python demo/text/text_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/seq2label_basic_Imdb \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 \
  --vocab_file=~/demo/data/IMDB/imdb_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --lr_method=linear_plus_warmup \
  tune_args \
  --train_dataset_meta=~/demo/data/IMDB/train.csv \
  --eval_dataset_meta=~/demo/data/IMDB/test.csv \
  --tune_config=source/tool/seq2label_basic_IMDB_tune_coarse.yaml


.. _seq2label_basic_export:

Export
---------------------------

::

  python demo/text/text_classification.py \
  --mode=export \
  --model_dir=~/demo/model/seq2label_basic_Imdb \
  --network=seq2label_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/data/IMDB/imdb_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --lr_method=linear_plus_warmup \
  export_args \
  --dataset_meta=~/demo/data/IMDB/train_clean.csv \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_text,input_mask \
  --output_ops=output_probabilities


.. _seq2label_basic_serve:

Serve
---------------------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_textclassification \
  --mount type=bind,source=/home/chuan/demo/model/seq2label_basic_Imdb/export,target=/models/textclassification \
  -e MODEL_NAME=textclassification -t tensorflow/serving:latest-gpu &

  python client/text_classification_client.py \
  --vocab_file=~/demo/data/IMDB/imdb_word_basic.vocab \
  --vocab_format=pickle \
  --vocab_top_k=40000 \
  --encode_method=basic