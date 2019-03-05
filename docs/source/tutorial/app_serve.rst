Model Serving
========================================

* :ref:`modelserving_installdocker`
* :ref:`modelserving_installnvidiadocker`
* :ref:`modelserving_serve`

.. _modelserving_installdocker:

Install Docker (Ubuntu 18.04)
----------------------------------------------

::

  sudo apt-get update

  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

  sudo add-apt-repository \
     "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) \
     stable"

  sudo apt-get update

  sudo apt-get install docker-ce=5:18.09.2~3-0~ubuntu-bionic

  sudo groupadd docker
  sudo usermod -aG docker $USER

.. _modelserving_installnvidiadocker:

Install Nvidia Docker
----------------------------------------------

::

  # If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
  docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
  sudo apt-get purge -y nvidia-docker

  # Add the package repositories
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update

  # Install nvidia-docker2 and reload the Docker daemon configuration
  sudo apt-get install -y nvidia-docker2
  sudo pkill -SIGHUP dockerd

  # Test nvidia-smi with the latest official CUDA image
  docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi

.. _modelserving_serve:

Serve
----------------------------------------------

The following three steps are used to serve the trained model:

* **Export**: The first step is to export the model as a ProtoBuffer file. For example, this is how to export a pre-trained resnet32 model for image classification:

::

  python demo/image/image_classification.py \
  --mode=export \
  --model_dir=~/demo/model/cifar10-resnet32-20180824 \
  --network=resnet32 \
  --augmenter=cifar_augmenter \
  --gpu_count=1 --batch_size_per_gpu=1 --epochs=1 \
  export_args \
  --export_dir=export \
  --export_version=1 \
  --input_ops=input_image \
  --output_ops=output_classes


More examples can be found here: `Image Segmentation <fcn.html#export>`__ , `Object Detection <ssd.html#export>`__, `Style Transfer <fns.html#export>`__, `Text Generation <app_text_generation.html#export>`__, `Text Classification <app_text_classification.html#export>`__.

* **Run TF-Serving**. A typical example of serving the exported model is like this:

::

  docker run --runtime=nvidia -p 8501:8501 \
  --name tfserving_classification \
  --mount type=bind,source=path_to_model_dir/export,target=/models/classification \
  -e MODEL_NAME=classification -t tensorflow/serving:latest-gpu &


* **Run client**. To consume the service, we use a client. For example, for image classification we run the client with this command:

::

  python client/image_classification_client.py --image_path=path_to_image
