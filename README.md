Quick Start
===

__Add Lambda deep learning demo to Python path__
```
export PYTHONPATH=$PYTHONPATH:path_to_lambda-deep-learning-demo
```

__Image Classification__
```
Train:
python demo/image_classification.py \
--num_gpu=4

Evaluation:
python demo/image_classification.py --mode=eval \
--num_gpu=4 --epochs=1 \
--dataset_meta=~/demo/data/cifar10/eval.csv

Infer:
python demo/image_classification.py --mode=infer \
--num_gpu=1 --batch_size_per_gpu=1 --epochs=1 \
--test_samples=~/demo/data/cifar10/test/appaloosa_s_001975.png,~/demo/data/cifar10/test/domestic_cat_s_001598.png,~/demo/data/cifar10/test/rhea_s_000225.png,~/demo/data/cifar10/test/trucking_rig_s_001216.png

Tune:
python demo/image_classification.py --mode=tune \
--num_gpu=4

Run with synthetic data:
Train
python demo/image_classification.py \
--inputter=image_classification_syn_inputter \
--augmenter="" \
--num_gpu=4

Evaluation
python demo/image_classification.py --mode=eval \
--inputter=image_classification_syn_inputter \
--augmenter="" \
--num_gpu=4 --epochs=1

Transfer Learning:
python demo/image_classification.py \
--mode=train \
--num_gpu=4 --epochs=20 --piecewise_boundaries=10 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/train.csv \
--model_dir=~/demo/model/image_classification_StanfordDog120 \
--pretrained_dir=~/demo/model/resnet_v2_50_2017_04_14 \
--skip_pretrained_var_list="resnet_v2_50/logits,global_step" \
--trainable_var_list="resnet_v2_50/logits"

python demo/image_classification.py \
--mode=eval \
--num_gpu=4 --epochs=1 \
--network=resnet50 \
--augmenter=vgg_augmenter \
--image_height=224 --image_width=224 --num_classes=120 \
--dataset_meta=~/demo/data/StanfordDogs120/eval.csv \
--model_dir=~/demo/model/image_classification_StanfordDog120
```

__Image Segmenation__
```
Train:
python demo/image_segmentation.py \
--num_gpu=4

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
--num_gpu=4
```

__Style Transfer__
```
Train:
python demo/style_transfer.py \
--num_gpu=4

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
--num_gpu=4
```