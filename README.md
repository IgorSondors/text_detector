### OCR-for-Russian-documents
#OCR application for recognition the Russian language text in documents

#Some dependencies and trot-helper

conda install -c anaconda tensorflow-gpu=1.14

conda install -c conda-forge matplotlib

conda install -c anaconda pillow

pip install Cython

pip install pycocotools

pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"


(tf_conda_gpu) C:\Users\sondors\Documents\TensorFlow\models\research\object_detection>python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config

python legacy/train.py --logtostderr \ --train_dir=training/ \ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config


#Для запуска eval параллельно с обучением:

CUDA_VISIBLE_DEVICES="" python legacy/eval.py \ --logtostderr \ --checkpoint_dir=training \ --pipeline_config_path=pack_detector/models/ssd_mobilenet_v1/ssd_mobilenet_v1_pack.config \ --eval_dir=test

tensorboard --logdir=C:\Users\sondors\Documents\TensorFlow\models\research\object_detection



#Запуск eval:

python legacy/eval.py \ --logtostderr \ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config \ --checkpoint_dir=training/ \ --eval_dir=eval/


#To visualize the eval results
tensorboard --logdir=eval/

#TO visualize the training results
tensorboard --logdir=training/

#To get pb-file of model

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-85000 --output_directory ssd_mobilenet_v1_coco\saved_model
