---
title: "How can I obtain label predictions from a trained CIFAR-10 model?"
date: "2025-01-30"
id: "how-can-i-obtain-label-predictions-from-a"
---
The key to obtaining label predictions from a trained CIFAR-10 model lies in understanding the output structure of the model after its training phase and the necessary steps to process input data into a format the model can understand. Having worked extensively with image classification models, specifically on embedded systems with limited resources, I've developed a streamlined approach for this. The core concept involves passing preprocessed image data through the model, extracting the raw output, and then mapping those outputs to the respective class labels.

Initially, a CIFAR-10 model is trained to map 32x32 pixel color images to one of ten predefined classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The output of a trained model, typically, consists of a vector of ten values representing the "score" or "logit" for each class. These are raw, unnormalized values and do not directly correspond to probabilities. A common strategy is to apply the softmax function to these values to obtain probabilities summing to one. However, for determining the predicted label, we usually only require the index of the highest score.

The typical workflow involves these key steps:

1. **Data Loading and Preprocessing**: The input images must be loaded and transformed into a format compatible with the model’s input layer. For CIFAR-10 models, this frequently involves converting images to a NumPy array with a specific shape (typically [batch_size, height, width, channels], e.g. [1, 32, 32, 3]) and pixel values scaled, often normalized to the range [0, 1] or [-1,1] depending on the specific training regime.
2. **Inference**: The preprocessed image data is then fed into the model. This results in the output vector containing raw scores for each class.
3. **Post-Processing**: To get the predicted label, we identify the index associated with the highest score in the output vector. This index corresponds to the class label.

Let me illustrate this with three code examples utilizing common machine learning libraries: TensorFlow/Keras and PyTorch.

**Example 1: TensorFlow/Keras**

```python
import tensorflow as tf
import numpy as np

# Load the trained model
model_path = "path/to/your/cifar10_model.h5"  # Replace with the actual path
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to make prediction
def predict_label(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    return predicted_class_index

# Example usage
image_path = "path/to/your/image.png"  # Replace with the actual path
predicted_index = predict_label(image_path)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
predicted_label = class_names[predicted_index]

print(f"The predicted label is: {predicted_label} (index: {predicted_index})")
```

In this example, I use TensorFlow and Keras. The model is loaded, an image is preprocessed to match the model’s input requirements, and then we perform inference. `np.argmax` determines the class index associated with the highest score. The normalization (`img_array / 255.0`) scales pixel values between 0 and 1 which is common in Keras CIFAR10 models. This is crucial for achieving accurate predictions, as deviations can significantly alter the model’s behaviour.  The `expand_dims` adds a batch dimension as the model expects batches as an input.

**Example 2: PyTorch**

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load the trained model
model_path = "path/to/your/cifar10_model.pth" # Replace with the actual path
model = torch.load(model_path)
model.eval() # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalization to [-1, 1]
])

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB") # Ensure image is RGB
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
    return img_tensor

# Function to make prediction
def predict_label(image_path):
    with torch.no_grad():  # Disable gradient calculations for inference
        input_tensor = preprocess_image(image_path)
        outputs = model(input_tensor)
        _, predicted_class_index = torch.max(outputs, 1)
    return predicted_class_index.item()


# Example Usage
image_path = "path/to/your/image.png" # Replace with the actual path
predicted_index = predict_label(image_path)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
predicted_label = class_names[predicted_index]
print(f"The predicted label is: {predicted_label} (index: {predicted_index})")
```

Here, I demonstrate a similar process using PyTorch. Notice the different approach to normalization via `transforms.Normalize`. PyTorch often employs normalization to the range [-1, 1]. The key difference from the Tensorflow/Keras example is the explicit use of `torch.no_grad()` within the prediction function, which optimizes memory usage and execution speed during inference. `torch.max` is used to find the index of the maximum value, and `.item()` to extract a single value from the tensor for the predicted index. The images are also opened using PIL to be compatible with TorchVision transforms. Also, it is critical to put the model in evaluation mode using `.eval()` to disable certain operations that are needed in training but not inference.

**Example 3: Handling Batch Predictions**

```python
import tensorflow as tf
import numpy as np
import os

# Load the trained model
model_path = "path/to/your/cifar10_model.h5"  # Replace with actual path
model = tf.keras.models.load_model(model_path)

# Function to preprocess a list of images
def preprocess_batch(image_paths):
    image_arrays = []
    for path in image_paths:
        img = tf.keras.utils.load_img(path, target_size=(32, 32))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0
        image_arrays.append(img_array)
    return np.array(image_arrays)

# Function to make predictions for a batch
def predict_batch_labels(image_paths):
    processed_images = preprocess_batch(image_paths)
    predictions = model.predict(processed_images)
    predicted_class_indices = np.argmax(predictions, axis=1)
    return predicted_class_indices

# Example usage with multiple images
image_dir = "path/to/image/folder" # Replace with the actual folder
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

predicted_indices = predict_batch_labels(image_files)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

for i, index in enumerate(predicted_indices):
  predicted_label = class_names[index]
  print(f"Image: {image_files[i]}, Predicted Label: {predicted_label} (index: {index})")
```

This example demonstrates batch processing, essential when dealing with multiple images. Instead of preprocessing one image at a time, I preprocess a list of image paths and stack them into a NumPy array. This is significantly more efficient than making individual predictions for each image.  The `np.argmax` now operates across `axis=1`, returning an array of the indices for each image in the batch, instead of just one index. The script also has a basic check to ensure only compatible image types are read.

For further in-depth understanding of these concepts and libraries, consider the official documentation for TensorFlow and PyTorch. Specifically, the Keras documentation within TensorFlow for model loading and image processing, and the torchvision documentation within PyTorch for image transforms, provide valuable details. Additionally, research and familiarity with the NumPy library, crucial for numerical computation in these workflows, are recommended. Further exploration of topics such as gradient descent, convolutional neural networks and the softmax function will provide a more holistic understanding of how to interpret the model's output.
