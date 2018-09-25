Transfer Learning
========================================

* :ref:`resnet50`
* :ref:`inceptionv4`
* :ref:`nasnetalarge`

.. _resnet50:

**ResNet50 on Stanford Dogs Dataset**
----------------------------------------------


Train with weights restored from pre-trained model

::

  python demo/image_classification.py \
  --mode=train \
  --model_dir=~/demo/model/resnet50_StanfordDogs120 \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
  --network=resnet50 \
  --augmenter=vgg_augmenter \
  --gpu_count=4 --batch_size_per_gpu=16 --epochs=50 \
  --num_classes=120 --image_width=224 --image_height=224 \
  train_args \
  --learning_rate=0.01 --optimizer=momentum \
  --piecewise_boundaries=25 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/StanfordDogs120/train.csv \
  --pretrained_model=~/demo/model/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt \
  --skip_pretrained_var=resnet_v2_50/logits,global_step \
  --trainable_vars=resnet_v2_50/logits

Hyper-Parameter Tuning

::

  python demo/image_classification.py \
  --mode=tune \
  --model_dir=~/demo/model/resnet50_StanfordDogs120 \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
  --network=resnet50 \
  --augmenter=vgg_augmenter \
  --gpu_count=4 --batch_size_per_gpu=16 --epochs=50 \
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

  python demo/image_classification.py \
  --mode=eval \
  --model_dir=~/demo/model/resnet50_StanfordDogs120 \
  --network=resnet50 \
  --augmenter=vgg_augmenter \
  --gpu_count=4 --batch_size_per_gpu=16 --epochs=1 \
  --num_classes=120 --image_width=224 --image_height=224 \
  eval_args \
  --dataset_meta=~/demo/data/StanfordDogs120/train.csv


.. _inceptionv4:

**InceptionV4 on Stanford Dogs Dataset**
----------------------------------------------


.. _nasnetalarge:

**NasNet-A-Large on Stanford Dogs Dataset**
----------------------------------------------