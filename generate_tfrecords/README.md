![](https://github.com/IgorSondors/text_detector/blob/master/generate_tfrecords/structure.jpeg)

### How to generate tfrecords

- It is necessary to collect data and do some preprocessing with it for real case similarity purpose

- Dataset should be labelled with one of the annotators app. [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html) is suitable

- Labels should download as csv files and modify those as this

class| fileName| height|width|xmax|	xmin|	ymax|	ymin|	text
 ---| ---| ---| ---| ---| ---| ---| ---| ---
rect|	1.jpg|	31|	326|	611|	285|	117|	86|	25 ОТДЕЛОМ МИЛИЦИИ


- Create Train folder for files mentioned in Train.csv, copy files there. Do the same for Test folder similarly
