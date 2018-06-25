Let us review the part of the code that perform the object detection and extract the images:


detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3new.jpg"), extract_detected_objects=True)

for eachObject, eachObjectPath in zip(detections, objects_path):
print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
print("Object's image saved in " + eachObjectPath)
print("--------------------------------")

In the above above lines, we called the detectObjectsFromImage() , parse in the input image path, output image part, and an extra parameter extract_detected_objects=True. This parameter states that the function should extract each object detected from the image and save it has a seperate image. The parameter is false by default. Once set to true, the function will create a directory which is the output image path + "-objects" . Then it saves all the extracted images into this new directory with each image's name being the detected object name + "-" + a number which corresponds to the order at which the objects were detected. 

This new parameter we set to extract and save detected objects as an image will make the function to return 2 values. The first is the array of dictionaries with each dictionary corresponding to a detected object. The second is an array of the paths to the saved images of each object detected and extracted, and they are arranged in order at which the objects are in the first array.




And one important feature you need to know!
You will recall that the percentage probability for each detected object is sent back by the detectObjectsFromImage() function. The function has a parameter minimum_percentage_probability , whose default value is 50 (value ranges between 0 - 100) . That means the function will only return a detected object if it's percentage probability is 50 or above. The value was kept at this number to ensure the integrity of the detection results. However, a number of objects might be skipped in the process. Therefore, you fine-tune the object detection by setting minimum_percentage_probability equal to a smaller value to detect more number of objects or higher value to detect less number of objects.