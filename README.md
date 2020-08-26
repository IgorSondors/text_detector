# Build dependencies

### Tensorflow Object Detection API

- clone [this](https://github.com/tensorflow/models.git) repository

- use [protobuf](https://developers.google.com/protocol-buffers/) for converting proto files to python scripts inside [protos folder](https://github.com/tensorflow/models/tree/master/research/object_detection/protos)

- download the [model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) you chose

- [download](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) config file for your model, change hyperparameters and write all paths inside it

- create training folder inside [object_detection folder](https://github.com/tensorflow/models/tree/master/research/object_detection) fill it with config file

- generate [tfrecords](https://github.com/IgorSondors/OCR-for-Russian-documents/tree/master/generate_tfrecords) and fill [data folder](https://github.com/tensorflow/models/tree/master/research/object_detection/data) with dataset (in the tfrecord format) and labels (csv files)

### Conda

- conda install -c anaconda tensorflow-gpu=1.14

- conda install -c conda-forge matplotlib

- conda install -c anaconda pillow

- pip install Cython

- pip install pycocotools

- pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

### Clean pip is better

- pip install tensorflow

- pip install tensorflow-gpu

- pip install Cython

- pip install pycocotools

- pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

- pip install pandas

Follow [this](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781) instructions for building CUDA and Cudnn dependencies if you use Windows

### Cheats

- cd C:\Users\sondors\Documents\TensorFlow\models\research\object_detection

- python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config

#### Train EfficentDet

- python model_main_tf2.py --pipeline_config_path=training\ssd_efficientdet_d3_896x896.config --model_dir=training --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr

- python model_main.py --model_dir=train --pipeline_config_path=training/ssd_efficientdet_d3_896x896.config --alsologtostderr --num_train_steps=80000 --num_eval_steps=1000

#### Для запуска eval параллельно с обучением:

- CUDA_VISIBLE_DEVICES="" python legacy/eval.py \ --logtostderr \ --checkpoint_dir=training \ --pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config \ --eval_dir=test

- tensorboard --logdir=C:\Users\sondors\Documents\TensorFlow\models\research\object_detection

#### Запуск eval:

- python legacy/eval.py \ --logtostderr \ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config \ --checkpoint_dir=training/ \ --eval_dir=eval/


#### To visualize the eval results
- tensorboard --logdir=eval/

#### TO visualize the training results
- tensorboard --logdir=training/

#### To get pb-file of model

- python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-85000 --output_directory ssd_mobilenet_v1_coco\saved_model

# Models that is ready for use

### Faster_RCNN

Download the model via [link](https://drive.google.com/file/d/1LFpO1DsDm2EHcYFPWQfAikgnHQ3mNPGm/view?usp=sharing)

### Mobilenet SSD

You can chose the model inside of this repository. Good model is [here](https://github.com/IgorSondors/OCR-for-Russian-documents/blob/master/frozen_inference_graph.pb)

### SSD EfficentDet

In progress
