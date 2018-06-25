Custom Object Detection
The object detection model (RetinaNet) supported by ImageAI can detect 80 different types of objects. They include: 
      person,   bicycle,   car,   motorcycle,   airplane,
          bus,   train,   truck,   boat,   traffic light,   fire hydrant,   stop_sign,
          parking meter,   bench,   bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra,
          giraffe,   backpack,   umbrella,   handbag,   tie,   suitcase,   frisbee,   skis,   snowboard,
          sports ball,   kite,   baseball bat,   baseball glove,   skateboard,   surfboard,   tennis racket,
          bottle,   wine glass,   cup,   fork,   knife,   spoon,   bowl,   banana,   apple,   sandwich,   orange,
          broccoli,   carrot,   hot dog,   pizza,   donot,   cake,   chair,   couch,   potted plant,   bed,
          dining table,   toilet,   tv,   laptop,   mouse,   remote,   keyboard,   cell phone,   microwave,
          oven,   toaster,   sink,   refrigerator,   book,   clock,   vase,   scissors,   teddy bear,   hair dryer,
          toothbrush.
Interestingly, ImageAI allow you to perform detection for one or more of the items above. That means you can customize the type of object(s) you want to be detected in the image. Let's take a look at the code below: 



custom_objects = detector.CustomObjects(car=True, motorcycle=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3custom.jpg"))

In the above code, after loading the model (can be done before loading the model as well), we defined a new variable "custom_objects = detector.CustomObjects()", in which we set its car and motorcycle properties equal to True. This is to tell the model to detect only the object we set to True. Then we call the "detector.detectCustomObjectsFromImage()" which is the function that allows us to perform detection of custom objects. Then we will set the "custom_objects" value to the custom objects variable we defined. 