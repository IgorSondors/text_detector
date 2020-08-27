#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application for text detection and cropping of text areas from passport photo to the server_crops folder
"""
import tensorflow as tf
import numpy as np
import cv2
import time
import os

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CLOUD_SERVER = r'C:\Users\sondors\Desktop\neurohives\cloud_server'
#PATH_TO_GRAPH = r'E:\Хранилище\TensorFlow_1.15\models1\research\object_detection\saved_model_dynamic\saved_model'#Faster RCNN
PATH_TO_GRAPH = r'C:\Users\sondors\Desktop\neurohives\cloud_server\neurohives'#efficientdet

load_graph_time = time.time()
detection_graph = tf.saved_model.load(PATH_TO_GRAPH)
print('load_graph_time is', time.time() - load_graph_time)

def run_inference_for_single_image(img, detection_graph):
  image = np.asarray(img)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = detection_graph.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def Y_coord_sorting(img, detection_graph):
    """
    Sorting areas of text corresponding Y axe
    :param img: 
    :param detection_graph:
    :variable text_coord: sorted detection_boxes
    :variable probability_list: sorted detection_scores
    :return: text_coord, probability_list, img_shape
    """
    output_dict = run_inference_for_single_image(img, detection_graph)
                 
    # Below extract rectangles coordinates    
    Xmax =[]
    Xmin =[]
    Ymax =[]
    Ymin =[]     
    probability_list = [] 
    
    for i, box in enumerate(output_dict['detection_boxes']):
        if output_dict['detection_scores'][i] > 0.30:   #ignore bounding boxes with probabilities lower than 0.30

            img_shape = np.shape(img)
            h, w = img_shape[:2]
            
            y1 = int(box[0]*h)
            x1 = int(box[1]*w)
            y2 = int(box[2]*h)
            x2 = int(box[3]*w)
                        
            deltaX = 0.1    # Add horizontal padding 10% of bounding box width
            deltaY = 0.2    # Add vertical padding 20% of bounding box height
            
            y1new = round(int(y1 - (y2 - y1)*deltaY))
            x1new = round(int(x1 - (x2 - x1)*deltaX))
            y2new = round(int(y2 + (y2 - y1)*deltaY))
            x2new = round(int(x2 + (x2 - x1)*deltaX))
            probability_list.append(output_dict['detection_scores'][i])
            Xmax.append(x2new)
            Xmin.append(x1new)
            Ymax.append(y2new)
            Ymin.append(y1new)

    text_coord = []    
    for i in range(len(Xmin)):
        text_coord.append([Xmin[i], Ymin[i], Xmax[i], Ymax[i]])     
    text_coord = sorted(text_coord, key=lambda r:r[3])

    return text_coord, probability_list, img_shape

def IOU(text_coord, probability_list):
    """
    Intersections over union with probability selection of almost the same areas with text
    :param text_coord: 
    :param probability_list: 
    :return: text_coord, probability_list
    """
    i = 1
    
    while i < len(text_coord): 
        last_rec = i-1
     
        if text_coord[last_rec][3] > text_coord[i][1] and text_coord[last_rec][3] < text_coord[i][3]:
            if text_coord[last_rec][0] > text_coord[i][0] and text_coord[last_rec][0] < text_coord[i][2] or text_coord[last_rec][2] > text_coord[i][0] and text_coord[last_rec][2] < text_coord[i][2] or text_coord[last_rec][0] < text_coord[i][0] and text_coord[last_rec][2] > text_coord[i][2] or text_coord[last_rec][0] > text_coord[i][0] and text_coord[last_rec][2] < text_coord[i][2]:
            
                text_coord_IOU = [max(text_coord[last_rec][0], text_coord[i][0]), max(text_coord[last_rec][1], text_coord[i][1]), min(text_coord[last_rec][2], text_coord[i][2]), min(text_coord[last_rec][3], text_coord[i][3])]

                width_last_rec = text_coord[last_rec][2] - text_coord[last_rec][0]
                heigh_last_rec = text_coord[last_rec][3] - text_coord[last_rec][1]
                S_last_rec = width_last_rec * heigh_last_rec

                width_i = text_coord[i][2] - text_coord[i][0]
                heigh_i = text_coord[i][3] - text_coord[i][1]
                S_i = width_i * heigh_i

                width_IOU = text_coord_IOU[2] - text_coord_IOU[0]
                heigh_IOU = text_coord_IOU[3] - text_coord_IOU[1]
                S_IOU = width_IOU * heigh_IOU

                if S_IOU/(S_i + S_last_rec) > 0.2:

                    if probability_list[last_rec] > probability_list[i]:
                        print('Detector stage removing', text_coord[i], probability_list[i])
                        text_coord.remove(text_coord[i])
                        probability_list.remove(probability_list[i])
                    elif probability_list[last_rec] < probability_list[i]:
                        print('Detector stage removing', text_coord[last_rec], probability_list[last_rec])
                        text_coord.remove(text_coord[last_rec])
                        probability_list.remove(probability_list[last_rec])
        
        if i == len(text_coord)-1:
            break
        i = i+1
    return text_coord, probability_list

def text_areas_cropping(img):
    """
    Cropping of sorted and cleared areas of text in the server_crops folder
    :param img: 
    :return: text_coord, img_shape, probability_list
    """        
    text_coord, probability_list, img_shape = Y_coord_sorting(img, detection_graph)
    text_coord, probability_list = IOU(text_coord, probability_list)
    
    Height = []
    Width = []
    for i in range(len(text_coord)):       
        Width.append(text_coord[i][2] - text_coord[i][0])
        Height.append(text_coord[i][3] - text_coord[i][1])

        pts1 = np.float32([[text_coord[i][0], text_coord[i][1]], [text_coord[i][2], text_coord[i][1]], [text_coord[i][2], text_coord[i][3]], [text_coord[i][0], text_coord[i][3]]])
        pts2 = np.float32([[0,0],[Width[i],0],[Width[i],Height[i]],[0,Height[i]]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(Width[i], Height[i]))

        cv2.imwrite(os.path.join(PATH_TO_CLOUD_SERVER,'server_crops\{}.jpg'.format(i)), dst)
    
    return text_coord, img_shape, probability_list

img_path = r'C:\\Users\\sondors\\Desktop\\neurohives\\cloud_server\\test_im\\1.jpg'
img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)

detection_time = time.time()
text_areas_cropping(img)
print('detection_time is', time.time() - detection_time)