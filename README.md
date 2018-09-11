Quick Start
===

__Add Lambda deep learning demo to Python path__
```
export PYTHONPATH=$PYTHONPATH:path_to_lambda-deep-learning-demo
```

__Image Classification__
```
Train:
python demo/image_classification.py --mode=train \
--num_gpu=4 --batch_size_per_gpu=256 --epochs=100 \
--piecewise_boundaries=50,75,90 --piecewise_learning_rate_decay=1.0,0.1,0.01,0.001 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10.tar.gz \
--dataset_meta=~/demo/data/cifar10/train.csv \
--model_dir=~/demo/model/image_classification_cifar10

Evaluation:
python demo/image_classification.py --mode=eval \
--num_gpu=4 --batch_size_per_gpu=256 --epochs=1 \
--dataset_meta=~/demo/data/cifar10/eval.csv \
--model_dir=~/demo/model/image_classification_cifar10

Infer:
python demo/image_classification.py --mode=infer \
--num_gpu=1 --batch_size_per_gpu=1 --epochs=1 \
--model_dir=~/demo/model/image_classification_cifar10 \
--test_samples=~/demo/data/cifar10/test/appaloosa_s_001975.png,~/demo/data/cifar10/test/domestic_cat_s_001598.png,~/demo/data/cifar10/test/rhea_s_000225.png,~/demo/data/cifar10/test/trucking_rig_s_001216.png

Tune:
python demo/image_classification.py --mode=tune \
--model_dir=~/demo/model/image_classification_cifar10 \
--num_gpu=4

Pre-trained Model:
curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10-resnet32-20180824.tar.gz | tar xvz -C ~/demo/model

python demo/image_classification.py --mode=eval \
--num_gpu=4 --batch_size_per_gpu=256 --epochs=1 \
--augmenter_speed_mode \
--dataset_meta=~/demo/data/cifar10/eval.csv \
--model_dir=~/demo/model/cifar10-resnet32-20180824
```

__Image Segmenation__
```
Train:
python demo/image_segmentation.py \
--num_gpu=4 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz

Evaluation:
python demo/image_segmentation.py --mode=eval \
--num_gpu=4 --epochs=1 \
--dataset_meta=~/demo/data/camvid/val.csv


Infer:
python demo/image_segmentation.py --mode=infer \
--batch_size_per_gpu=1 --epochs=1 --num_gpu=1 \
--test_samples=~/demo/data/camvid/test/0001TP_008550.png,~/demo/data/camvid/test/Seq05VD_f02760.png,~/demo/data/camvid/test/Seq05VD_f04650.png,~/demo/data/camvid/test/Seq05VD_f05100.png

Tune:
python demo/image_segmentation.py --mode=tune \
--num_gpu=1
```

__Style Transfer__
```
python demo/style_transfer.py \
--num_gpu=4 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/mscoco_fns.tar.gz

Eval:
python demo/style_transfer.py --mode=eval \
--num_gpu=4 --epochs=1 \
--dataset_meta=~/demo/data/mscoco_fns/eval2014.csv

Infer:
python demo/style_transfer.py --mode=infer \
--batch_size_per_gpu=1 --epochs=1 --num_gpu=1 \
--test_samples=~/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg

Tune:
python demo/style_transfer.py --mode=tune \
--num_gpu=1
```