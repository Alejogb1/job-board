---
title: "yolov8 predict model object detection?"
date: "2024-12-13"
id: "yolov8-predict-model-object-detection"
---

Okay so you're looking at YOLOv8 and how to use its prediction capabilities for object detection right Been there done that a few times trust me I've wrestled with this thing more than I care to admit

Alright lets break it down real simple

First off you gotta have your YOLOv8 model trained or at least one of the pretrained models readily available and yes I know the process and you know the drill but just in case you got a brand new shiny YOLOv8 model it won't do much until you feed it some data or you know the pre-trained ones so thats kinda an important point right so let's assume you have a valid model if you don't go back and train it and we can talk later when you have that crucial part nailed down okay? Cool

Now lets say your model is a file called `best.pt` its just the most common name for a trained model lets get to the meat of it for predictions.  The easiest way for predictions is using the Python API and you’re probably using that anyways if you are here lets get to it.

```python
from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Run inference on an image
results = model.predict(source="image.jpg")

# Print the results
for result in results:
    boxes = result.boxes  # Boxes object for bounding box coordinates
    for box in boxes:
        print(box.xyxy)  # Print bounding box coordinates in xyxy format
        print(box.conf) #confidence scores of the object detections
        print(box.cls) #class indexes detected
```

Alright there we have it the most basic prediction example you will ever see I like to keep it simple and it works for all sorts of models you know the ones in the Ultralytics ecosystem which I always recommend for a quick deployment and testing. So what’s going on here?

First we import the `YOLO` class from the `ultralytics` library which you need to have installed I hope you do if you are doing this otherwise install the ultralytics package you probably need to upgrade it to get the latest features anyway. Then we create a `model` object by loading a pretrained YOLO model or you know the custom one you trained which i'm pretty sure you have because you are here asking about predictions so i'm sure you did the training part which is probably the most important of the whole process.

Then we call the `predict` method this takes one argument the path to an image `image.jpg` which I know that you probably have some images already. Then it gives us back a `results` object which I iterate over. Inside that you access the `boxes` attribute then you can iterate through each one of them. And each box has the bounding box coordinates and confidence scores and class ids.

Now let's say you wanna display this on your screen we'll need to add a bit of code for that you know a bit more of work.

```python
from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("best.pt")

# Run inference on an image
results = model.predict(source="image.jpg")

# Load the image
image = cv2.imread("image.jpg")

# Loop over the results
for result in results:
    boxes = result.boxes  # Boxes object for bounding box coordinates
    for box in boxes:
        xyxy = box.xyxy[0].tolist() # convert to list for display
        conf = box.conf.item()  # Get the confidence score
        cls = int(box.cls.item())  # Get the class ID
        x1,y1,x2,y2 = map(int, xyxy) # convert to integers for drawing the rectangle
        label = f"{model.names[cls]}: {conf:.2f}" # build the label for the box
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2) #draw the rectangle
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) #draw the text label

cv2.imshow("Object Detection", image) # Show the annotated image
cv2.waitKey(0) #wait until a key is pressed to close the window
cv2.destroyAllWindows() # close all windows
```

Here we are using `cv2` which is OpenCV library, you should already have it installed but if you don't you should do so using `pip install opencv-python` and again I expect you to already have it but its always good to give the proper instructions. We read the image and then we start looping through the boxes and extracting the xyxy coordinates, the confidence score and the class id. Then we extract the names from the model using the `model.names` and build the label string and use `cv2` to draw the bounding boxes and the labels for each detection and finally display the annotated image.

Now sometimes you might have multiple images in a folder and you wanna batch process them no problem we can do that also. We will just need to adapt the previous example a bit.

```python
from ultralytics import YOLO
import cv2
import os

# Load the model
model = YOLO("best.pt")

# Input and output directories
input_dir = "input_images"
output_dir = "output_images"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_dir, filename)
        # Run inference on an image
        results = model.predict(source=image_path)
        # Load the image
        image = cv2.imread(image_path)
        # Loop over the results
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box coordinates
            for box in boxes:
                xyxy = box.xyxy[0].tolist() # convert to list for display
                conf = box.conf.item()  # Get the confidence score
                cls = int(box.cls.item())  # Get the class ID
                x1,y1,x2,y2 = map(int, xyxy) # convert to integers for drawing the rectangle
                label = f"{model.names[cls]}: {conf:.2f}" # build the label for the box
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2) #draw the rectangle
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) #draw the text label

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image) # save the annotated image

print(f"Images saved to {output_dir}")
```

We introduce two new concepts here an input directory and an output directory where the annotated images will go. And the rest of the process is pretty much the same. We loop over each image in a directory and then perform the predictions and display bounding boxes and labels and save them in an output folder. And of course we are making sure we are not processing any files that are not images or that dont have an appropriate extension you know a bit of error checking is always welcome.

Now about the model and stuff. In the examples i am using a file called `best.pt` this file might be different for you. You might need to modify that to reflect your particular file name it’s like saying ‘best’ is a relative concept you know?. But lets be serious here I think you already know that.

Also sometimes you might get that output in different formats like you want the bounding boxes in a different formats like instead of x1,y1,x2,y2 you want the center and the width and the height I understand that pain I used to spend so much time converting between them now there is the bounding box object and it’s attributes that provide such formats too `box.xywh` `box.xyxy` and so on. So you dont need to do manual conversions anymore but if you still prefer to do those you know the math it’s simple. I’m a strong advocate for using what the framework gives you as far as possible.

And now let's get serious for a moment I've seen folks get tripped up on the version compatibility of the ultralytics library I had a project once where I spent hours debugging a simple inference script only to find out I was using an older version of Ultralytics that didn’t play nice with the newer model I trained it was not fun I had to manually go back all my code and it was not fun believe me so always double-check that you are using the latest version you can and always check the changelogs to see what's changed its a life saver. Also be very aware of your CUDA drivers as they are essential for optimal performance if your GPU is not working properly you should check that too.

Also if you are deep into this you should totally check the documentation for the ultralytics library they have been getting much better lately also papers are the way to go for some more advanced things like understanding how the detection really works from a math perspective.

For books on the subject look for classic ones like "Deep Learning" by Goodfellow and also look for recent books that dive into computer vision with deep learning and of course the original papers that explain how the YOLO models work and evolve. You can start with the original YOLO paper then move to YOLOv2 YOLOv3 YOLOv4 all the way to YOLOv8 and beyond you know it's a long journey but totally worth it. It also helps you to understand how things work on the lower levels and it will make you a better developer.

And always always test your code thoroughly it’s something you need to do even if you are 100% sure it's working because bugs are usually very subtle so you need to check all edge cases possible believe me I have made that mistake myself. So always check even the simple parts of the code you have to be meticulous you know?

Anyways hope this gives you a good starting point if you need anything else just let me know and I'll try to help.
