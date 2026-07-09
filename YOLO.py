from ultralytics import YOLO
import cv2
import numpy as np
import os
model_path = os.path.join( 'models', 'YOLO.pt')
model = YOLO(model_path)


def convertImageCV(image):
    image_array = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    return image_array
def yoloCrop(image):
    predict=model(image)[0]
    boxes=predict.boxes
    boxData=boxes.data
    crops=[]
    for box in boxData:
        crops.append({"x":box[0],"y":box[1],"width":box[2]-box[0],"height":box[3]-box[1]})
    return crops

