## Комманды для обучения детектора

#### Все команды следует вводить в терминале открытого из папки object_detection

```python
cd C:\Users\sondors\Documents\TensorFlow\models\research\object_detection
```
### TF1

#### Пример обучения на TF1 бэкенде (Mobilenet)

```python
python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
```

#### Запуск evaluation:

```python
python legacy/eval.py \ --logtostderr \ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config \ --checkpoint_dir=training/ \ --eval_dir=eval/
```

#### Визуализация работы детектора на валидационных данных

```python
tensorboard --logdir=eval/
```

#### Визуализация работы детектора на тренировочных данных

```python
tensorboard --logdir=training/
```

#### Для получения frozen_inference_graph.pb файла с замороженными весами сети пригодного для TF1 модели

```python
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-85000 --output_directory ssd_mobilenet_v1_coco\saved_model
```

### TF2

#### Пример обучения на TF2 бэкенде (EfficentDet)

```python
python model_main_tf2.py --pipeline_config_path=training\ssd_efficientdet_d3_896x896.config --model_dir=training --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr
```
```python
python model_main.py --model_dir=train --pipeline_config_path=training/ssd_efficientdet_d3_896x896.config --alsologtostderr --num_train_steps=80000 --num_eval_steps=1000
```

#### Для получения saved_model.pb файла с весами сети пригодного для TF2 модели

```python
python exporter_main_v2.py \ --input_type image_tensor \ --pipeline_config_path training/ssd_efficientdet_d3_896x896.config \ --trained_checkpoint_dir training \ --output_directory efficientdet_d3_coco17_tpu-32/saved_mode
```

#### Для использования исскуственного расширения данных

Смотрите [этот](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) репозиторий

## Необходимые требования для использования комманд для тренировки сети

### 1) Создать виртуальное окружение используя conda или pip

#### Conda + pip (простая установка)

- Введите следующие команды в CMD/terminal

```
conda install -c anaconda tensorflow-gpu=1.15
```
```
conda install -c conda-forge matplotlib
```
```
conda install -c anaconda pillow
```
```
pip install Cython
```
```
pip install pycocotools
```
```
pip install pandas
```
```
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

#### Чистый pip (лучше, но сложнее в установке)

- Введите следующие команды в CMD/terminal

```
pip install tensorflow
```
```
pip install tensorflow-gpu
```
```
pip install Cython
```
```
pip install pycocotools
```
```
pip install pandas
```
```
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```
- Следуйте [этим](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781) инструкциям для построения CUDA и Cudnn зависимостей, если вы используете Windows или смотрите официальную документацию [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). При использовании Ubuntu нужная ссылка [another CUDA documentation]([CUDA Toolkit documentation]) вместо предыдущих

Для проверки возможности запуска кода на GPU в вашем виртуальном окружении запустите [эту программу](https://github.com/IgorSondors/cv-trash/blob/master/TFcheck.py)

### 2) Tensorflow Object Detection API

- Клонируйте [этот](https://github.com/tensorflow/models.git) репозиторий
```
git clone https://github.com/tensorflow/models.git
```

- Используйте [protobuf](https://developers.google.com/protocol-buffers/) для конвертации proto файлов в python скрипты внутри [protos папки](https://github.com/tensorflow/models/tree/master/research/object_detection/protos)
```python
protoc object_detection/protos/*.proto --python_out=.
```
- Загрузите [model TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) или [model TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) в зависимости от вашего выбора бэкенда

- [download](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) config file for your model, change hyperparameters and write all paths inside it

- create training folder inside [object_detection folder](https://github.com/tensorflow/models/tree/master/research/object_detection) fill it with config file

- [generate tfrecords](https://github.com/IgorSondors/OCR-for-Russian-documents/tree/master/generate_tfrecords) and fill [data folder](https://github.com/tensorflow/models/tree/master/research/object_detection/data) with dataset (in the tfrecord format)

You can check [this tutorial](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73), if you have difficulties with using text_detector repo
Now you are ready for training process. Check "Commands for training" section above

# Models that is ready for use

Instead of training your custom text_detector model you can use one of few models below 

### Faster_RCNN

Download the weights via [link](https://drive.google.com/file/d/1LFpO1DsDm2EHcYFPWQfAikgnHQ3mNPGm/view?usp=sharing). This model was trained with [this](https://github.com/IgorSondors/text_detector/blob/master/faster_rcnn_resnet101_coco.config) configs for 32k global steps. Saved_model and weights for fine tuning via [link](https://drive.google.com/file/d/1K4k6xtebwUEpAQeSHLPx1m3MynHhQHjZ/view?usp=sharing)

### Mobilenet SSD

You can chose the model inside of this repository. Good model weights is [here](https://github.com/IgorSondors/OCR-for-Russian-documents/blob/master/frozen_inference_graph.pb).

### SSD EfficentDet

Download the weights via [link](https://drive.google.com/drive/folders/13J2zvihwHqyxmsTtWH4QU_9b5r9oxNL5?usp=sharing)
