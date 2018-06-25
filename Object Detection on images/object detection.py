# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 06:52:08 2018

@author: hp
"""

from imageai.Detection import ObjectDetection
import os
import numpy as np
import cv2

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\object detection\\resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\object detection\\input images\\bus.jpeg"), output_image_path=os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\object detection\\output images\\imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )

