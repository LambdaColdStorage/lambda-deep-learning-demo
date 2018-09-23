Image Segmenation
-------------------------------------

::

  python demo/image_classification.py \
  --mode=train \
  --model_dir=~/demo/model/image_classification_cifar10 \
  --dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=128 --epochs=100 \
  train_args \
  --learning_rate=0.5 --optimizer=momentum \
  --piecewise_boundaries=75 \
  --piecewise_lr_decay=1.0,0.1 \
  --dataset_meta=~/demo/data/cifar10/train.csv