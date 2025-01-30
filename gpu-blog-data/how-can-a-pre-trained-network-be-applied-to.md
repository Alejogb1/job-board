---
title: "How can a pre-trained network be applied to a larger image using Keras/TensorFlow?"
date: "2025-01-30"
id: "how-can-a-pre-trained-network-be-applied-to"
---
A common challenge encountered when deploying convolutional neural networks (CNNs) trained on smaller input sizes is their application to larger, real-world images. The fixed input size of the network, determined during training, often clashes with the variable dimensions of data in practical scenarios. Effectively using a pre-trained model on larger images necessitates understanding how convolutional layers operate and leveraging techniques such as sliding window prediction or fully convolutional network adaptation.

Specifically, a CNN’s convolutional layers perform local feature extraction, using filters to identify patterns within receptive fields. The subsequent pooling layers downsample the spatial dimensions of the feature maps. These convolutional operations inherently apply to any input image exceeding the initial training size, as they perform localized calculations irrespective of the overall image dimensions. The limitation arises from the fully connected layers, typically present in the latter stages of a CNN, designed to convert the final feature maps into a fixed-size vector suitable for classification or regression tasks. These fully connected layers require a consistent number of inputs, directly tied to the dimensions of the flattened feature maps coming from preceding convolutional and pooling operations.

The crucial step in applying a pre-trained network to a larger image is to eliminate these fixed-size fully connected layers and replace them with equivalent convolutional layers. This approach allows the network to output a feature map (or a set of class activation maps) instead of a single vector of predictions. This adaptation enables us to interpret the output of the network at multiple spatial locations in the larger image, essentially generating a heatmap of activations instead of a single classification. This output can then be interpreted using techniques like sliding windows, where predictions are made by sliding a window of a size compatible with the pre-trained model across the input image, or, as described above, interpreting the output activations map directly.

Consider the following example, where I’ve used a pre-trained VGG16 model, initially trained on 224x224 images. Instead of directly applying the model to a larger 448x448 image, which would result in an error due to the discrepancy in input size, I’ll first modify the model architecture and then demonstrate two application scenarios.

**Code Example 1: Adapting the Pre-trained VGG16 Model to a Fully Convolutional Network (FCN)**

This code demonstrates how to replace the fully connected layers of a pre-trained VGG16 model with convolutional layers. I assume familiarity with the Keras functional API.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model

def adapt_vgg16_to_fcn():
    base_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    # Remove the fully connected layers by taking the output of the last pooling layer
    base_output = base_model.get_layer('block5_pool').output

    # Define the new input layer to accept arbitrary size images
    input_tensor = Input(shape=(None, None, 3))

    # Create a new model using the same convolutional layers of VGG16
    x = base_model.get_layer('block1_conv1')(input_tensor)
    x = base_model.get_layer('block1_conv2')(x)
    x = base_model.get_layer('block1_pool')(x)
    x = base_model.get_layer('block2_conv1')(x)
    x = base_model.get_layer('block2_conv2')(x)
    x = base_model.get_layer('block2_pool')(x)
    x = base_model.get_layer('block3_conv1')(x)
    x = base_model.get_layer('block3_conv2')(x)
    x = base_model.get_layer('block3_conv3')(x)
    x = base_model.get_layer('block3_pool')(x)
    x = base_model.get_layer('block4_conv1')(x)
    x = base_model.get_layer('block4_conv2')(x)
    x = base_model.get_layer('block4_conv3')(x)
    x = base_model.get_layer('block4_pool')(x)
    x = base_model.get_layer('block5_conv1')(x)
    x = base_model.get_layer('block5_conv2')(x)
    x = base_model.get_layer('block5_conv3')(x)
    x = base_model.get_layer('block5_pool')(x)

    # Add equivalent Conv2D layers to replace fully connected layers
    x = Conv2D(filters=4096, kernel_size=(7, 7), padding='valid', activation='relu', name="fc1")(x)
    x = Conv2D(filters=4096, kernel_size=(1, 1), padding='valid', activation='relu', name="fc2")(x)
    x = Conv2D(filters=1000, kernel_size=(1, 1), padding='valid', activation='softmax', name="predictions")(x)

    fcn_model = Model(inputs=input_tensor, outputs=x)

    # Transfer the trained weights
    for i in range(len(base_model.layers)-4):
      fcn_model.layers[i+1].set_weights(base_model.layers[i+1].get_weights())
    
    # Copy weights from the old fully connected layers (fc1, fc2)
    fc1_weights_old = base_model.get_layer("fc1").get_weights()
    fc1_weights_new = [tf.reshape(fc1_weights_old[0], (7,7,512,4096)), fc1_weights_old[1]]
    fcn_model.get_layer("fc1").set_weights(fc1_weights_new)

    fc2_weights_old = base_model.get_layer("fc2").get_weights()
    fc2_weights_new = [tf.reshape(fc2_weights_old[0], (1,1,4096,4096)), fc2_weights_old[1]]
    fcn_model.get_layer("fc2").set_weights(fc2_weights_new)


    predictions_weights_old = base_model.get_layer("predictions").get_weights()
    predictions_weights_new = [tf.reshape(predictions_weights_old[0], (1,1,4096,1000)), predictions_weights_old[1]]
    fcn_model.get_layer("predictions").set_weights(predictions_weights_new)
    return fcn_model
```

This function, `adapt_vgg16_to_fcn`, creates a fully convolutional VGG16 by removing the original fully connected layers and adding equivalent convolutional layers, which retain the same number of filters and preserve the learned weights by reshaping them appropriately. The input layer is also modified to accept images of arbitrary size. The weights of these new convolutional layers are initialized based on the weights of original fully connected layers.  This allows for applying the learned filters in a spatially aware way across the entire input image.

**Code Example 2: Applying the FCN with Sliding Windows using the Modified Model**

This example simulates applying the network with sliding windows. In practice you’d use more robust methods for efficiency, but this demonstrates the concept.

```python
import numpy as np
from PIL import Image

def apply_fcn_with_sliding_window(model, image_path, window_size=(224, 224), stride=112):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    height, width, _ = image.shape

    output_maps = np.zeros((height, width, 1000)) # Assuming 1000 output classes

    for y in range(0, height - window_size[0], stride):
        for x in range(0, width - window_size[1], stride):
            window = image[y:y + window_size[0], x:x + window_size[1]]
            window = np.expand_dims(window, axis=0) # Add batch dim
            window = window / 255.0 # Normalize
            predictions = model.predict(window)
            output_maps[y:y+window_size[0], x:x+window_size[1], :] = predictions[0]
    return output_maps

# Assume fcn_model from previous code example
fcn_model = adapt_vgg16_to_fcn()
image_path = "large_image.jpg"  # Replace with your image path
output_maps = apply_fcn_with_sliding_window(fcn_model, image_path)
```

This `apply_fcn_with_sliding_window` function demonstrates the process of using the adapted FCN model with a sliding window approach. I take an input image, divide it into overlapping windows of a certain size (matching the size the original model was trained on) , apply the FCN, and store the predictions at the locations on a full-sized output map. This approach simulates scanning the whole image with the pre-trained model in a local way. This method is computationally expensive for large images because of the overlapping windows.

**Code Example 3: Direct Application of FCN output map to a Large Image**

This example demonstrates the more efficient method of directly processing the large input image by the fully convolutional model. It relies on the fact that the output of the adapted network is a spatial map of activations, which can then be resized to the original image size.

```python
from tensorflow.keras.layers import Resizing

def apply_fcn_direct(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    height, width, _ = image.shape
    
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalise

    feature_maps = model.predict(image)
    output_map_size = feature_maps.shape[1:3]
    resizer = Resizing(height, width)
    resized_feature_maps = resizer(feature_maps[0]).numpy()
    return resized_feature_maps

# Assume fcn_model from previous code example
fcn_model = adapt_vgg16_to_fcn()
image_path = "large_image.jpg"  # Replace with your image path
output_maps_direct = apply_fcn_direct(fcn_model, image_path)
```

The function `apply_fcn_direct` shows the direct application of the fully convolutional network. It takes an image, expands its dimensions to be compatible with the network, performs predictions with the model, then resizes the predicted output map back to the original image resolution using a resizing layer.  The `Resizing` layer can use various interpolation methods and is critical for upscaling the low-resolution feature maps. This method leverages the FCN's ability to produce an activation map based on its input size, without requiring manual sliding window techniques.

Implementing these methods requires attention to detail and an understanding of the underlying model architecture and image processing. The chosen technique should consider the specific task’s nature and computational budget. A sliding window is computationally expensive but preserves finer-grained details, while directly upsampling the output may result in loss of detail due to interpolation.

For further understanding, I recommend consulting resources on convolutional neural networks, specifically focusing on fully convolutional networks, the functional API of Keras and TensorFlow and image segmentation/object detection strategies. Exploration of model architectures like U-Net, which are explicitly designed for pixel-wise output, is beneficial. Reviewing code examples for image segmentation will additionally provide concrete application examples beyond generic class prediction.
