![](https://github.com/IgorSondors/text_detector/blob/master/generate_tfrecords/structure.jpeg)

### How to generate tfrecords

- It is necessary to collect data and do some preprocessing with it for real case similarity purpose

- Dataset should be labelled with one of the annotators app. [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html) is suitable

- Labels should download as csv files and modify those as this

class| fileName| height|width|xmax|	xmin|	ymax|	ymin|	text
 ---| ---| ---| ---| ---| ---| ---| ---| ---
rect|	1.jpg|	31|	326|	611|	285|	117|	86|	25 ОТДЕЛОМ МИЛИЦИИ

- [It](https://github.com/IgorSondors/cv-trash/blob/master/static_str_delete.py) might help to clear csv

- Create Train folder for files mentioned in Train.csv, copy files there. Do the same for Test folder similarly

- Execute [this](https://github.com/IgorSondors/text_detector/blob/master/generate_tfrecords/generate_tfrecord.py) script with using TF1.* or [this](https://github.com/IgorSondors/text_detector/blob/master/generate_tfrecords/TF2_generate_tfrecord.py) using TF2.*

- Move received tfrecords and [class_file](https://github.com/IgorSondors/text_detector/blob/master/object-detection.pbtxt) to [data](https://github.com/tensorflow/models/tree/master/research/object_detection/data) folder
