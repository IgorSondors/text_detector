![](https://github.com/IgorSondors/text_detector/blob/master/generate_tfrecords/structure.jpeg)

### Генерация tfrecords

- Небходимо собрать данные и сделать их похожими на те, что будут встречаться в реальной работе детектора. Для препроцессинга данных, в частности перспективного преобразования используйте [эту](https://github.com/IgorSondors/CV-preprocessing/blob/master/warp_4clics.py) программу

- Датасет нужно разметить, для этого может подойти приложение [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html)

- Разметку стоит перевести в csv формат и привести ее к следующему виду

class| fileName| height|width|xmax|	xmin|	ymax|	ymin|	text
 ---| ---| ---| ---| ---| ---| ---| ---| ---
rect|	1.jpg|	31|	326|	611|	285|	117|	86|	25 ОТДЕЛОМ МИЛИЦИИ

- [Эта](https://github.com/IgorSondors/cv-trash/blob/master/static_str_delete.py) программа может помочь почистить csv файл от статических строк

- Датасет нужно разделить на Train.csv и Test.csv, создать папки Train и Test, в которые далее переместить размеченные картинки согласно присутствию в соответствующем csv файле. Рекомендованное разделение на Train/Test - 90/10, в виду малого количества данных

- Для создания tfrecords используйте [этот](https://github.com/IgorSondors/text_detector/blob/master/generate_tfrecords/generate_tfrecord.py) скрипт для TF1.* бэкенда или [этот](https://github.com/IgorSondors/text_detector/blob/master/generate_tfrecords/TF2_generate_tfrecord.py) для TF2.*

- Переместите полученные tfrecords и [файл класса](https://github.com/IgorSondors/text_detector/blob/master/object-detection.pbtxt) в папку [data](https://github.com/tensorflow/models/tree/master/research/object_detection/data)
