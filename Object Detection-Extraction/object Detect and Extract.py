# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 08:40:28 2018

@author: hp
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI object Detection3\\resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI object Detection3\\input images\\image3.jpg"), output_image_path=os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI object Detection3\\output images\\image3new.jpg"), extract_detected_objects=True)


for eachObject, eachObjectPath in zip(detections, objects_path):
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
    print("Object's image saved in " + eachObjectPath)
print("--------------------------------")

