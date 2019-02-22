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

Evaluation

::

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
  --vocab_file=/home/ubuntu/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  eval_args \
  --dataset_meta=/home/ubuntu/demo/data/IMDB/test_clean.csv


Infer

::

  python demo/text_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/seq2label_basic_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/data/IMDB/vocab_basic.txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  infer_args \
  --callbacks=infer_basic,infer_display_text_classification \
  --test_samples="This movie is awesome ."#"This movie is bad ."#"This movie has an unusual taste ."#"It is not clear what this movie is about ."#"This is not a very good movie ."#"I saw this at the premier at TIFF and was thrilled to learn the story is about a real friendship . This is not a typical road movie , or buddy film . Given the lead actors , I knew it would be something special , and it is . Entertaining , funny in parts , hard to accept in others - as a white american who was not around in the 1960's , the racism was mind boggling and I could not help but feel shame . Green Book has so many layers - family , culture , honesty , dignity , genius , respect , acceptance , stereotypes , racism , music , class , friendship , and fried chicken . Whatever your views , race , or age - this film is not 'preachy' , but you should appreciate an honest portrayal of a difficult time & place in history . I'll use the term an unlikely friendship , but knowing the two men were real makes it fantastic . I'm so grateful to have learned about them and their lives . I only wish there had been a Q&A afterward ." \
  --splitter=#


  python demo/text_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/seq2label_glove_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_basic \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/model/glove.6B/glove.6B.200d.txt \
  --vocab_top_k=40000 \
  --encode_method=basic \
  infer_args \
  --callbacks=infer_basic,infer_display_text_classification \
  --test_samples="This movie is awesome ."#"This movie is bad ."#"This movie has an unusual taste ."#"It is not clear what this movie is about ."#"This is not a very good movie ."#"I saw this at the premier at TIFF and was thrilled to learn the story is about a real friendship . This is not a typical road movie , or buddy film . Given the lead actors , I knew it would be something special , and it is . Entertaining , funny in parts , hard to accept in others - as a white american who was not around in the 1960's , the racism was mind boggling and I could not help but feel shame . Green Book has so many layers - family , culture , honesty , dignity , genius , respect , acceptance , stereotypes , racism , music , class , friendship , and fried chicken . Whatever your views , race , or age - this film is not 'preachy' , but you should appreciate an honest portrayal of a difficult time & place in history . I'll use the term an unlikely friendship , but knowing the two men were real makes it fantastic . I'm so grateful to have learned about them and their lives . I only wish there had been a Q&A afterward ." \
  --splitter=#


  python demo/text_classification.py \
  --mode=infer \
  --model_dir=~/demo/model/seq2label_bert_Imdb \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/temp.tar.gz \
  --network=seq2label_bert \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  --vocab_file=/home/ubuntu/demo/model/uncased_L-12_H-768_A-12/vocab.txt \
  --vocab_top_k=-1 \
  --encode_method=bert \
  infer_args \
  --callbacks=infer_basic,infer_display_text_classification \
  --test_samples="This movie is awesome ."#"This movie is bad ."#"This movie has an unusual taste ."#"It is not clear what this movie is about ."#"This is not a very good movie ."#"I saw this at the premier at TIFF and was thrilled to learn the story is about a real friendship . This is not a typical road movie , or buddy film . Given the lead actors , I knew it would be something special , and it is . Entertaining , funny in parts , hard to accept in others - as a white american who was not around in the 1960's , the racism was mind boggling and I could not help but feel shame . Green Book has so many layers - family , culture , honesty , dignity , genius , respect , acceptance , stereotypes , racism , music , class , friendship , and fried chicken . Whatever your views , race , or age - this film is not 'preachy' , but you should appreciate an honest portrayal of a difficult time & place in history . I'll use the term an unlikely friendship , but knowing the two men were real makes it fantastic . I'm so grateful to have learned about them and their lives . I only wish there had been a Q&A afterward ." \
  --splitter=#

Hyper-Parameter Tuning

::



**Export**
------------

::