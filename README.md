Lambda Deep Learning Demos
===

Welcome to Lambda Lab's deep learning demo suite -- the place to find ready-to-use machine learnig models. We offer the following cool features:

* A curate of open-source, state-of-the-art models that cover major machine learning applications, including image classification, image segmentation, object detection, image style transfer, text classification and generation etc.

* Pure Tensorflow implementation. Efforts are made to keep the boilplate consistent across all demos.

* Examples of transfer learning and how to adapt the model to new data.

* Model serving

Check this [documents](https://lambda-deep-learning-demo.readthedocs.io/en/latest/) for details.


### Applications


---


#### Images Classification

| Model        | Dataset           | Top 1 Accuracy | Pre-trained Model  |
| ------------- |:-------------:| -----:|:-----:|
| ResNet32      | CIFAR10 |  92% |  [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/cifar10-resnet32-20180824.tar.gz) |
| ResNet50 Fine-Tune     | StanfordDogs    | 75.36%  | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/resnet50_StanfordDogs120-20190303.tar.gz)  |
| InceptionV4 Fine-Tune  | StanfordDogs      | 92.4% | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/inceptionv4_StanfordDogs120-20190306.tar.gz)  |
| NasNet-A-Large Fine-Tune | StanfordDogs   |  94.99% | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/nasnet_A_large_StanfordDogs120-20190306.tar.gz)  |


---


#### Images Segmentation


| Model        | Dataset           | Accuracy | Pre-trained Model  |
| ------------- |:-------------:| -----:|:-----:|
| FCN      | CamVid |  86.6%  | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/fcn_camvid_20190125.tar.gz) |
| U-Net      | CamVid    | 86.9% |  [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/unet_camvid_20190125.tar.gz) |


---


#### Object Detection

| Model        | Dataset           | (AP) IoU=0.50:0.95 | Pre-trained Model  |
| ------------- |:-------------:| -----:|:-----:|
| SSD300      | MSCOCO | 21.9 |  [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/ssd300_mscoco_20190105.tar.gz) |
| SSD500     | MSCOCO    | 25.7 | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/ssd512_mscoco_20190105.tar.gz) |


---


#### Style Transfer

| Model        | Dataset           | Pre-trained Model  |
| ------------- |:-------------:|:-----:|
| Fast Neural Style      | MSCOCO | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/fns_gothic_20190126.tar.gz) |


---


#### Text Generation

| Model        | Dataset           | Pre-trained Model  |
| ------------- |:-------------:|:-----:|
| Char RNN      | Shakespeare | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/char_rnn_shakespeare-20190303.tar.gz)  |
| Word RNN      | Shakespeare | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/word_rnn_shakespeare-20190303.tar.gz)  |
| Word RNN  + Glove | Shakespeare | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/word_rnn_glove_shakespeare-20190303.tar.gz)  |


---



#### Text Classification

| Model        | Dataset           | Classification Accuracy  | Pre-trained Model  |
| ------------- |:-------------:| -----:|:-----:|
| LSTM      | IMDB |  85.2%  | [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/seq2label_basic_Imdb-20190303.tar.gz)  |
| LSTM + Glove      | IMDB | 86.1%  |  [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/seq2label_glove_Imdb-20190303.tar.gz) |
| Transfer Learning + BERT | IMDB |  92.2% |  [Download](https://s3-us-west-2.amazonaws.com/lambdalabs-files/seq2label_bert_Imdb-20190303.tar.gz) |

---

### Citation
If you use our code in your research or wish to refer to the examples, please cite with:

```
@misc{lambdalabs2018demo,
  title={Lambda Labs Deep Learning Demos},
  author={Lambda Labs, inc.},
  howpublished={\url{https://github.com/lambdal/lambda-deep-learning-demo}},
  year={2018}
}
```
