---
title: "find symbol from image ai vision?"
date: "2024-12-13"
id: "find-symbol-from-image-ai-vision"
---

Okay so you're looking to pluck symbols out of images using AI vision huh I get it Been there done that Seriously I've spent way too many nights wrestling with this exact problem back in my early days when I was trying to build a text recognition system for old scanned documents think ancient Sumerian cuneiform but in image form yeah it was a nightmare

Basically you're venturing into the realm of object detection and character recognition sometimes called Optical Character Recognition or OCR but in your case it's generalized to symbols not necessarily characters which is a bit more complex Let's break this down in a way that even someone relatively new to this can grasp I'll throw in some code examples in Python since that's what most folks use these days

First things first you need to locate the symbols in the image This is the object detection part It involves drawing bounding boxes around the symbols you want to extract We're talking about finding the regions of interest or ROIs as they're usually called This is where convolutional neural networks CNNs come into play specifically models trained for object detection

A popular choice is the YOLO family You Only Look Once It's efficient for real-time processing and surprisingly good at locating even small objects I remember spending weeks trying to get it working properly on a Raspberry Pi for a robotics project it was like pulling teeth seriously

Here's a basic example using a pre-trained YOLO model with a library called OpenCV which is pretty much the standard for computer vision stuff in Python

```python
import cv2
import numpy as np

def detect_symbols(image_path):
  net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") #these are the usualy yolov3 weights

  classes = []
  with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

  layer_names = net.getLayerNames()
  output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

  img = cv2.imread(image_path)
  height, width, channels = img.shape

  blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
  net.setInput(blob)
  outs = net.forward(output_layers)

  class_ids = []
  confidences = []
  boxes = []

  for out in outs:
    for detection in out:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > 0.5:  #confidence threshold here you can tweak it
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #Non Max Supression which is crucial

  for i in range(len(boxes)):
    if i in indexes:
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  cv2.imshow("Detected Symbols", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


detect_symbols("image.jpg")
```
This example assumes you have the YOLOv3 weights `yolov3.weights` configuration `yolov3.cfg` and class names file `coco.names` ready to go You need to download them online they are readily available Don't worry too much if you don't understand every line of code right now Just know that it takes an image as input and tries to detect objects from the coco dataset like person car etc If it detects any object the code draws bounding boxes around the detected objects showing them visually The core logic is in there

Now this is generic object detection You need a model that's trained on your specific symbols This is where transfer learning comes in handy

Basically you take a pre-trained model like the one above that already knows a lot about detecting general objects and fine-tune it on your own symbol dataset This requires collecting a large set of images of your symbols ideally with variations in lighting orientation and scale The more data the better that's one of the most hard to learn lesson for every programmer I learned it the hard way

For fine-tuning you'll need a framework like TensorFlow or PyTorch They're more complex than just using OpenCV but they're indispensable for serious AI work

Here is a simple example on how to use pytorch to classify some object it's not object detection like the yolo example before but it will get you going in the fine-tuning direction I'm trying to show that you need to fine-tune on your specific data.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform

        classes = os.listdir(data_dir)
        for idx, cls in enumerate(classes):
            class_dir = os.path.join(data_dir, cls)
            for image_name in os.listdir(class_dir):
              self.image_paths.append(os.path.join(class_dir,image_name))
              self.labels.append(idx)
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
data_dir = "symbol_dataset" # replace this for your data structure 
train_dataset = CustomDataset(os.path.join(data_dir,"train"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(os.path.join(data_dir,"val"), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a basic CNN model for this example you can use resnet or any other model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Define model, loss and optimizer
num_classes = len(os.listdir(os.path.join(data_dir,"train"))) #number of symbol to detect
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    #Validation step
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
             outputs = model(images)
             _,predicted=torch.max(outputs.data,1)
             total += labels.size(0)
             correct += (predicted==labels).sum().item()
    print('Validation Accuracy of the model on the validation images: {} %'.format(100*correct/total))
    

print('Finished Training')
```
This is a basic image classification example not object detection so do not expect to detect any image but it will get you started on the pytorch side The idea is that you need to use a classifier to recognize the bounding boxes extracted from the first step

Now once you have the bounding boxes you need to recognize what the symbol inside each bounding box is This is where the actual classification comes in

This will need anther model trained on your specific symbols but with only one symbol in each bounding box If you have a large dataset you might use a state of the art model architecture like ResNet or MobileNet and fine tune it This part is often called the recognition or classification part

Here is an example using TensorFlow

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# Define parameters
image_size = (128, 128)
batch_size = 32

# Load datasets
data_dir = "symbol_dataset"  # Replace this with your dataset directory
train_dataset = image_dataset_from_directory(
    os.path.join(data_dir,"train"),
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

val_dataset = image_dataset_from_directory(
    os.path.join(data_dir,"val"),
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)
# Define CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_dataset.class_names), activation='softmax') # Number of classes equal to total number of symbols
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset)

# Evaluate the model
evaluation_results = model.evaluate(val_dataset)
print(f"Validation Loss: {evaluation_results[0]:.4f}")
print(f"Validation Accuracy: {evaluation_results[1]:.4f}")
```
This code is similar to the pytorch code but now using tensorflow

Now about the resources you asked for Instead of giving random links I'd suggest checking out these books

*   **"Deep Learning with Python" by François Chollet**  This is a great starting point for deep learning in general especially if you're using Keras with TensorFlow

*   **"Computer Vision: Algorithms and Applications" by Richard Szeliski** This is more of an academic dive but it covers everything from the basics to very advanced computer vision techniques

*   **"Hands-On Machine Learning with Scikit-Learn Keras & TensorFlow" by Aurélien Géron** This will help with the practical side of machine learning including preprocessing pipelines and stuff like that

And remember building a good symbol recognition system takes time and experimentation Don't get discouraged by the initial results or if your cat steals your data labels They're more complicated than it might seem initially I've been debugging computer vision problems for years and still sometimes I'm just like what just happened and why is my bounding box off by 3 pixels

Oh one more thing if you see some weird behavior in your model it's usually not aliens it's just the optimizer is not optimal
Good luck with your project and let me know if I can help you with anything else
