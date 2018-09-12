Quick Start
===

__Image Classification__
```
Resnet32

Train:
python demo/image_classification.py --mode=train \
--num_gpu=4 --batch_size_per_gpu=256 --epochs=100 \
--piecewise_boundaries=50,75,90 \
--piecewise_learning_rate_decay=1.0,0.1,0.01,0.001 \
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
--dataset_meta=~/demo/data/cifar10/train.csv \
--model_dir=~/demo/model/image_classification_cifar10 \
--num_gpu=4

Pre-trained Model:
curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10-resnet32-20180824.tar.gz | tar xvz -C ~/demo/model

python demo/image_classification.py --mode=eval \
--num_gpu=4 --batch_size_per_gpu=256 --epochs=1 \
--augmenter_speed_mode \
--dataset_meta=~/demo/data/cifar10/eval.csv \
--model_dir=~/demo/model/cifar10-resnet32-20180824

Transfer Learning:
(mkdir ~/demo/model/resnet_v2_50_2017_04_14;
curl http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz | tar xvz -C ~/demo/model/resnet_v2_50_2017_04_14)

python demo/image_classification.py --mode=train \
--num_gpu=4 --batch_size_per_gpu=64 --epochs=20 \
--piecewise_boundaries=10 \
--piecewise_learning_rate_decay=1.0,0.1 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/train.csv \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/StanfordDogs120.tar.gz \
--model_dir=~/demo/model/image_classification_StanfordDog120 \
--pretrained_dir=~/demo/model/resnet_v2_50_2017_04_14 \
--skip_pretrained_var_list="resnet_v2_50/logits,global_step" \
--trainable_var_list="resnet_v2_50/logits"

python demo/image_classification.py \
--mode=eval \
--num_gpu=4 --batch_size_per_gpu=64 --epochs=1 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/eval.csv \
--model_dir=~/demo/model/image_classification_StanfordDog120

Train with synthetic data:
python demo/image_classification.py \
--mode=train \
--num_gpu=4 --batch_size_per_gpu=64 --epochs=1000 --piecewise_boundaries=10 \
--network=resnet50 \
--inputter=image_classification_syn_inputter \
--augmenter="" \
--image_height=224 --image_width=224 --num_classes=120 \
--model_dir=~/demo/model/image_classification_StanfordDog120
```

__Image Segmenation__
```
Train:
python demo/image_segmentation.py --mode=train \
--num_gpu=4 --batch_size_per_gpu=16 --epochs=200 \
--piecewise_boundaries=100 \
--piecewise_learning_rate_decay=1.0,0.1 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/camvid.tar.gz \
--dataset_meta=~/demo/data/camvid/train.csv \
--model_dir=~/demo/model/image_segmentation_camvid

Evaluation:
python demo/image_segmentation.py --mode=eval \
--num_gpu=4 --batch_size_per_gpu=16 --epochs=1 \
--dataset_meta=~/demo/data/camvid/val.csv \
--model_dir=~/demo/model/image_segmentation_camvid

Infer:
python demo/image_segmentation.py --mode=infer \
--batch_size_per_gpu=1 --epochs=1 --num_gpu=1 \
--model_dir=~/demo/model/image_segmentation_camvid \
--test_samples=~/demo/data/camvid/test/0001TP_008550.png,~/demo/data/camvid/test/Seq05VD_f02760.png,~/demo/data/camvid/test/Seq05VD_f04650.png,~/demo/data/camvid/test/Seq05VD_f05100.png

Tune:
python demo/image_segmentation.py --mode=tune \
--dataset_meta=~/demo/data/camvid/train.csv \
--model_dir=~/demo/model/image_segmentation_camvid \
--num_gpu=4
```

__Style Transfer__
```
Train:
python demo/style_transfer.py --mode=train \
--num_gpu=4 --batch_size_per_gpu=4 --epochs=10 \
--piecewise_boundaries=5 \
--piecewise_learning_rate_decay=1.0,0.1 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/mscoco_fns.tar.gz \
--dataset_meta=~/demo/data/mscoco_fns/train2014.csv \
--model_dir=~/demo/model/style_transfer_mscoco_fns

Eval:
python demo/style_transfer.py --mode=eval \
--num_gpu=4 --batch_size_per_gpu=4 --epochs=1 \
--dataset_meta=~/demo/data/mscoco_fns/eval2014.csv \
--model_dir=~/demo/model/style_transfer_mscoco_fns

Infer:
python demo/style_transfer.py --mode=infer \
--batch_size_per_gpu=1 --epochs=1 --num_gpu=1 \
--model_dir=~/demo/model/style_transfer_mscoco_fns \
--test_samples=~/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,~/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg

Tune:
python demo/style_transfer.py --mode=tune \
--num_gpu=4 \
--dataset_meta=~/demo/data/mscoco_fns/train2014.csv \
--model_dir=~/demo/model/style_transfer_mscoco_fns
```

__Text Generation__

```
Train:
python demo/text_generation.py --mode=train \
--num_gpu=4 --batch_size_per_gpu=128 --epochs=1000 \
--piecewise_boundaries=500,750,900 \
--piecewise_learning_rate_decay=1.0,0.1,0.01,0.001 \
--dataset_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/shakespeare.tar.gz \
--dataset_meta=~/demo/data/shakespeare/shakespeare_input.txt \
--model_dir=~/demo/model/text_gen_shakespeare


Inference:
python demo/text_generation.py --mode=infer \
--num_gpu=1 --batch_size_per_gpu=1 --epochs=1 \
--model_dir=~/demo/model/text_gen_shakespeare
```