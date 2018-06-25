# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 08:48:36 2018

@author: hp
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI object Detection4\\resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
#detector.loadModel(detection_speed="fast")
#detector.loadModel(Detection Speed = "normal" , Minimum Percentage Probability = 50 (default), Detection Time = 63.5 seconds)
#detector.loadModels(Detection Speed = "fast" , Minimum Percentage Probability = 40 (default), Detection Time = 20.8 seconds)
#detector.loadModels(Detection Speed = "faster" , Minimum Percentage Probability = 30 (default), Detection Time = 11.2 seconds)
#detector.loadModels(Detection Speed = "fastest" , Minimum Percentage Probability = 30 (default), Detection Time = 7.6 seconds)
#detector.loadModels(Detection Speed = "flash" , Minimum Percentage Probability = 30 (default), Detection Time = 3.85 seconds)

#detector.loadModels(Detection Speed = "flash" , Minimum Percentage Probability = 10 (default), Detection Time = 3.67 seconds)






custom_objects = detector.CustomObjects(car=True, motorcycle=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI object Detection4\\input images\\image3.jpg"), output_image_path=os.path.join(execution_path , "C:\\Users\\hp\\Downloads\\imageAI object Detection4\\output images\\image3custom.jpg"))


for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
print("--------------------------------")