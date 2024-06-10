from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")          #YOLO Pretrained model

dir = os.path.dirname(os.path.realpath(__file__))       #os current directory path

dog_img = f"{dir}/Dog_img.jpg"                      #Dog Image Location
cats_img = f"{dir}/Cat_img.jpg"                     #Cat Image Location
cats_img = f"{dir}/cats.jpg"                        #Multiple Cats Image Location   (Cat count of 4)

results = model(dog_img)                            #Providing image path to pretrained YOLO model (Using YOLOv8n)
detect = results[0].boxes.data.tolist()             #Caching detected objects coco name (YOLO Pretrained model has coco list of different object)

cat_no = 15                                         #Cat count in coco list is 15.0
dog_no = 16                                         #Dog count in coco list is 16.0
count = len(results[0].boxes)                       #Getting count of object detected in image

if int(detect[0][-1]) == cat_no:                    #Checking detected object is Cat or dog using it coco number
    print(f"{count} cat detected")                  #Showing count of cat detected in image
elif int(detect[0][-1]) ==dog_no:
    print(f"{count} dog detected")                  #Showing count of dog detected in image