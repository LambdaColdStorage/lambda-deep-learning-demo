Transfer Learning
========================================

* :ref:`resnet50`
* :ref:`inceptionv4`
* :ref:`nasnetalarge`

.. _resnet50:

ResNet50 on Stanford Dogs Dataset
----------------------------------------------

Download Dataset

::

  python demo/download_data.py \
  --data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
  --data_dir=~/demo/data


Download Pre-trained Model

::

  (mkdir ~/demo/model/resnet_v2_50_2017_04_14;curl http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz | tar xvz -C ~/demo/model/resnet_v2_50_2017_04_14)


Train with weights restored from pre-trained model

::

  python demo/image/image_classification.py \
  --mode=train \
  --model_dir=~/demo/model/resnet50_StanfordDogs120 \
  --network=resnet50 \
  --augmenter=vgg_augmenter \
  --batch_size_per_gpu=16 --epochs=10 \
  --num_classes=120 --image_width=224 --image_height=224 \
  train_args \
  --learning_rate=0.1 --optimizer=momentum \
  --piecewise_boundaries=5 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/StanfordDogs120/train.csv \
  --pretrained_model=~/demo/model/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt \
  --skip_pretrained_var=resnet_v2_50/logits,global_step,power \
  --trainable_vars=resnet_v2_50/logits

Hyper-Parameter Tuning

::

  python demo/image/image_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/resnet50_StanfordDogs120 \
  --network=resnet50 \
  --augmenter=vgg_augmenter \
  --batch_size_per_gpu=16 \
  --num_classes=120 --image_width=224 --image_height=224 \
  tune_args \
  --train_dataset_meta=~/demo/data/StanfordDogs120/train.csv \
  --eval_dataset_meta=~/demo/data/StanfordDogs120/eval.csv \
  --pretrained_model=~/demo/model/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt \
  --skip_pretrained_var=resnet_v2_50/logits,global_step,power \
  --trainable_vars=resnet_v2_50/logits \
  --tune_config=source/tool/resnet50_stanforddogs120_tune_coarse.yaml

Evaluation

::

  python demo/image/image_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/resnet50_StanfordDogs120 \
  --network=resnet50 \
  --augmenter=vgg_augmenter \
  --batch_size_per_gpu=16 --epochs=1 \
  --num_classes=120 --image_width=224 --image_height=224 \
  eval_args \
  --dataset_meta=~/demo/data/StanfordDogs120/eval.csv


.. _inceptionv4:

InceptionV4 on Stanford Dogs Dataset
----------------------------------------------

Download pre-trained model

::

  (mkdir ~/demo/model/inception_v4_2016_09_09;curl http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz | tar xvz -C ~/demo/model/inception_v4_2016_09_09)

Train with weights restored from pre-trained model

::

  python demo/image/image_classification.py \
  --mode=train \
  --model_dir=~/demo/model/inceptionv4_StanfordDogs120 \
  --network=inception_v4 \
  --augmenter=inception_augmenter \
  --batch_size_per_gpu=16 --epochs=4 \
  --num_classes=120 --image_width=299 --image_height=299 \
  train_args \
  --learning_rate=0.1 --optimizer=momentum \
  --piecewise_boundaries=2 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/StanfordDogs120/train.csv \
  --pretrained_model=~/demo/model/inception_v4_2016_09_09/inception_v4.ckpt \
  --skip_pretrained_var=InceptionV4/AuxLogits,InceptionV4/Logits,global_step,power \
  --trainable_vars=InceptionV4/AuxLogits,InceptionV4/Logits

Hyper-Parameter Tuning

::

  python demo/image/image_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/inceptionv4_StanfordDogs120 \
  --network=inception_v4 \
  --augmenter=inception_augmenter \
  --batch_size_per_gpu=16 \
  --num_classes=120 --image_width=299 --image_height=299 \
  tune_args \
  --train_dataset_meta=~/demo/data/StanfordDogs120/train.csv \
  --eval_dataset_meta=~/demo/data/StanfordDogs120/eval.csv \
  --pretrained_model=~/demo/model/inception_v4_2016_09_09/inception_v4.ckpt \
  --skip_pretrained_var=InceptionV4/AuxLogits,InceptionV4/Logits,global_step,power \
  --trainable_vars=InceptionV4/AuxLogits,InceptionV4/Logits \
  --tune_config=source/tool/inceptionv4_stanforddogs120_tune_coarse.yaml

Evaluation

::

  python demo/image/image_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/inceptionv4_StanfordDogs120 \
  --network=inception_v4 \
  --augmenter=inception_augmenter \
  --batch_size_per_gpu=16 --epochs=1 \
  --num_classes=120 --image_width=299 --image_height=299 \
  eval_args \
  --dataset_meta=~/demo/data/StanfordDogs120/eval.csv

.. _nasnetalarge:

NasNet-A-Large on Stanford Dogs Dataset
----------------------------------------------

Download pre-trained model

::

  (mkdir ~/demo/model/nasnet-a_large_04_10_2017;curl https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz | tar xvz -C ~/demo/model/nasnet-a_large_04_10_2017)

Train with weights restored from pre-trained model

::

  python demo/image/image_classification.py \
  --mode=train \
  --model_dir=~/demo/model/nasnet_A_large_StanfordDogs120 \
  --network=nasnet_A_large \
  --augmenter=inception_augmenter \
  --batch_size_per_gpu=16 --epochs=4 \
  --num_classes=120 --image_width=331 --image_height=331 \
  train_args \
  --learning_rate=0.1 --optimizer=momentum \
  --piecewise_boundaries=2 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/StanfordDogs120/train.csv \
  --pretrained_model=~/demo/model/nasnet-a_large_04_10_2017/model.ckpt \
  --skip_pretrained_var=final_layer,aux_logits,global_step,power \
  --trainable_vars=final_layer,aux_logits

Hyper-Parameter Tuning

::

  python demo/image/image_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/nasnet_A_large_StanfordDogs120 \
  --network=nasnet_A_large \
  --augmenter=inception_augmenter \
  --batch_size_per_gpu=16 \
  --num_classes=120 --image_width=331 --image_height=331 \
  tune_args \
  --train_dataset_meta=~/demo/data/StanfordDogs120/train.csv \
  --eval_dataset_meta=~/demo/data/StanfordDogs120/eval.csv \
  --pretrained_model=~/demo/model/nasnet-a_large_04_10_2017/model.ckpt \
  --skip_pretrained_var=final_layer,aux_logits,global_step,power \
  --trainable_vars=final_layer,aux_logits \
  --tune_config=source/tool/nasnetalarge_stanforddogs120_tune_coarse.yaml

Evaluation

::

  python demo/image/image_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/nasnet_A_large_StanfordDogs120 \
  --network=nasnet_A_large \
  --augmenter=inception_augmenter \
  --batch_size_per_gpu=16 --epochs=1 \
  --num_classes=120 --image_width=331 --image_height=331 \
  eval_args \
  --dataset_meta=~/demo/data/StanfordDogs120/eval.csv