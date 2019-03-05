Sequence-to-label Glove
========================================


* :ref:`seq2label_glove_downloaddata`
* :ref:`seq2label_glove_preprocess`
* :ref:`seq2label_glove_downloadvocab`
* :ref:`seq2label_glove_train`
* :ref:`seq2label_glove_eval`
* :ref:`seq2label_glove_inference`
* :ref:`seq2label_glove_tune`
* :ref:`seq2label_glove_export`
* :ref:`seq2label_glove_serve`


.. _seq2label_glove_downloaddata:

Download Dataset
----------------------------------------------

::

  python demo/download_data.py \
  --data_url=http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
  --data_dir=~/demo/data/


.. _seq2label_glove_preprocess:

Preprocess Dataset
---------------------------------------------

::

  python demo/text/preprocess/preprocess_aclImdb_v1.py \
  --remove_punctuation=False


.. _seq2label_glove_downloadvocab:

Download Glove Embedding
----------------------------------------------

::

  wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip -d ~/demo/model/glove.6B && rm glove.6B.zip


.. _seq2label_glove_train:

Train from scratch
----------------------------------------------

::

  python demo/text/text_classification.py \
  --mode=train \
  --model_dir=~/demo/model/seq2label_glove_Imdb \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 --epochs=100 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  --lr_method=linear_plus_warmup \
  train_args \
  --learning_rate=0.002 --optimizer=adam \
  --dataset_meta=~/demo/data/IMDB/train.csv


.. _seq2label_glove_eval:

Evaluation
----------------------------------------------

::

  python demo/text/text_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/seq2label_glove_Imdb \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 --epochs=1 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --unit=word \
  eval_args \
  --dataset_meta=~/demo/data/IMDB/test.csv


.. _seq2label_glove_inference:

Inference
---------------------

::

  python demo/text/text_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/seq2label_glove_Imdb \
  --network=seq2label_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  infer_args \
  --callbacks=infer_basic,infer_display_text_classification \
  --test_samples="This movie is awesome."#"This movie is bad."#"This movie has an unusual taste."#"It is not clear what this movie is about."#"This is not a very good movie."#"I saw this at the premier at TIFF and was thrilled to learn the story is about a real friendship." \
  --splitter=#


.. _seq2label_glove_tune:

Hyper-Parameter Tuning
---------------------------------

::

  python demo/text/text_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/seq2label_glove_Imdb \
  --network=seq2label_basic \
  --batch_size_per_gpu=128 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --lr_method=linear_plus_warmup \
  tune_args \
  --train_dataset_meta=~/demo/data/IMDB/train.csv \
  --eval_dataset_meta=~/demo/data/IMDB/test.csv \
  --tune_config=source/tool/seq2label_glove_IMDB_tune_coarse.yaml

.. _seq2label_glove_export:

Export
---------------------------

::

  python demo/text/text_classification.py \
  --mode=export \
  --model_dir=~/demo/model/seq2label_glove_Imdb \
  --network=seq2label_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  --lr_method=linear_plus_warmup \
  export_args \
  --dataset_meta=~/demo/data/IMDB/train.csv \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_text,input_mask \
  --output_ops=output_probabilities

.. _seq2label_glove_serve:

Serve
---------------------------

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_textclassification \
  --mount type=bind,source=/home/chuan/demo/model/seq2label_glove_Imdb/export,target=/models/textclassification \
  -e MODEL_NAME=textclassification -t tensorflow/serving:latest-gpu &

  python client/text_classification_client.py \
  --vocab_file=~/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_format=txt \
  --vocab_top_k=40000 \
  --encode_method=basic