# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 08:14:28 2018

@author: hp
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI objectDetection1\\resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI objectDetection1\\input images\\image2.jpg"), output_image_path=os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI objectDetection1\\output images\\image2new.jpg"))


for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
print("--------------------------------")