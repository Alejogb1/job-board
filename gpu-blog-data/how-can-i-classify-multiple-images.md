---
title: "How can I classify multiple images?"
date: "2025-01-30"
id: "how-can-i-classify-multiple-images"
---
The classification of multiple images, a frequent task in computer vision, often requires a pipeline approach involving data loading, preprocessing, model inference, and result interpretation. I've encountered numerous implementations of this, from embedded systems processing sensor feeds to large-scale analysis of satellite imagery. Optimizing each of these steps for efficiency and accuracy is paramount.

Fundamentally, classifying multiple images is not significantly different from classifying a single image. The core difference lies in how we manage the batch of inputs and the subsequent processing of the results. Instead of feeding one image at a time to a model, we construct a data loader that generates batches, allowing for parallel processing and improved throughput, particularly when employing hardware acceleration such as GPUs.

Here's a breakdown of the process, incorporating specific practical experience:

**1. Data Loading:**

First, the images need to be read and organized. The most straightforward method would be reading image paths into a list. However, more complex real-world applications often involve loading data from databases, cloud storage, or sensor streams. I have frequently used Python's `os` module for basic directory traversal, but libraries like `tensorflow.data` or `torch.utils.data` become indispensable for large datasets, offering built-in functionalities for batching, shuffling, and prefetching data, significantly reducing bottlenecks. When dealing with terabytes of medical imaging data, efficient data loading was crucial for training even relatively simple classification models.

**2. Preprocessing:**

Raw images typically require preprocessing before being passed to the model. This involves steps such as resizing to a uniform shape, normalization (scaling pixel values to a specific range), and potentially data augmentation (applying random transformations like rotations, flips, and crops to increase the diversity of training data and improve generalization). For example, I recall a project involving handwritten digit recognition, where simple grayscale conversion and centering significantly improved the classifier’s performance compared to raw, unnormalized color images. Preprocessing is typically done in a batched manner alongside data loading to optimize input pipeline speed.

**3. Model Inference:**

The core task involves applying a pre-trained or newly trained classification model to our batch of images. Libraries such as TensorFlow, PyTorch, and scikit-learn provide models and methods for both model definition and inference. I have experimented with models ranging from small, custom convolutional neural networks (CNNs) for constrained devices to complex pre-trained architectures like ResNet or EfficientNet for complex visual tasks. Inference involves passing the preprocessed batches through the model, which yields probability distributions over classes for each image in the batch.

**4. Result Interpretation:**

After inference, we interpret the predicted probabilities and associate each image with a class label. This often involves selecting the class with the highest probability or thresholding to detect multi-label classification scenarios. I've used methods such as `argmax` in NumPy to extract the class label from the probability outputs. When analyzing multi-label satellite data, I needed to set class probability thresholds to avoid false positives and negatives based on model precision and recall.

Here are three code examples demonstrating the image classification process, employing popular Python libraries.

**Example 1: Simple Batch Inference with TensorFlow/Keras**

```python
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        images.append(img_array)
    return np.concatenate(images, axis=0) / 255.0


# Placeholder for image paths.
image_paths = [os.path.join("images", f) for f in os.listdir("images") if f.endswith(".jpg")]

# Load pre-trained model.
model = tf.keras.applications.MobileNetV2(weights='imagenet')


preprocessed_batch = load_and_preprocess_images(image_paths)
predictions = model.predict(preprocessed_batch)

# Decode the predictions.
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

for i, prediction in enumerate(decoded_predictions):
    print(f"Image {i+1}:")
    for label, desc, prob in prediction:
        print(f"    {desc}: {prob:.2f}")

```
*Commentary:* This code snippet uses `tensorflow.keras` for loading a pre-trained `MobileNetV2` model. The `load_and_preprocess_images` function takes a list of image paths, loads, and preprocesses them (resizes and normalizes). Then, it performs inference on the whole batch and decodes the predictions using the model's `decode_predictions` function. This approach is appropriate for initial investigations, small-scale testing, and projects where model building is not the focus. Note that error handling and path validation should be included in production code.

**Example 2: Custom Data Generator with PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import os
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx]).float() / 255.0
        label = self.labels[idx]
        if self.transform:
             image = self.transform(image)
        return image, label


# Placeholder data.
image_paths = [os.path.join("images", f) for f in os.listdir("images") if f.endswith(".jpg")]
labels = np.random.randint(0, 10, len(image_paths)) #replace with actual labels

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset = CustomImageDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define simple CNN model
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 64 * 64, 10) #10 output classes
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch_idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}")

```

*Commentary:* This snippet demonstrates the use of a custom PyTorch `Dataset` and `DataLoader` for image loading and preprocessing. It includes a basic `transform` pipeline and a rudimentary CNN for demonstration, performing training and displaying loss. I've employed similar structures when working on tasks like medical image segmentation. This structure is suitable when you require flexibility in data loading, have specific preprocessing requirements, or want to control batch generation more closely.

**Example 3: Scikit-learn with Feature Extraction**

```python
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
import numpy as np

def load_and_extract_features(image_paths, target_size=(100, 100)):
    features = []
    for path in image_paths:
        img = imread(path, as_gray=True) #convert to grayscale
        resized_img = resize(img, target_size)
        features.append(resized_img.flatten())
    return np.array(features)


# Placeholder image paths and labels.
image_paths = [os.path.join("images", f) for f in os.listdir("images") if f.endswith(".jpg")]
labels = np.random.randint(0, 2, len(image_paths))# Binary classification


features = load_and_extract_features(image_paths)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy {accuracy:.2f}")
```

*Commentary:* This example uses `scikit-learn` for image classification. It illustrates a simple process of loading images, converting them to grayscale, resizing, flattening them into a feature vector, and then training a `RandomForestClassifier`. I often use this approach for simple classification tasks where high accuracy is not critical, and quick experimentation is the priority. Feature extraction using methods beyond simple flattening can be combined with this approach.

For additional study, consult documentation for the TensorFlow, PyTorch, and Scikit-learn libraries. Seek information on image data loading best practices, convolutional neural networks, and different classification models. Further research into data augmentation techniques, batch normalization, and optimization methods will also improve the quality of image classification systems.
