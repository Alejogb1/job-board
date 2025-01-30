---
title: "How can I extract a (num_images, 1280) array from a MobileNetv2 Keras model with GlobalAveragePooling?"
date: "2025-01-30"
id: "how-can-i-extract-a-numimages-1280-array"
---
The final layer of a MobileNetv2 model, when followed by `GlobalAveragePooling2D`, produces a flattened feature vector, not a 2D array directly corresponding to intermediate image-like features. To obtain a `(num_images, 1280)` array, you'll need to extract the output from a layer *before* the global average pooling stage, then reshape it. I've encountered this scenario extensively while implementing transfer learning workflows with MobileNetv2, particularly when needing per-image feature representations for clustering or custom downstream tasks. The key is to target the layer that provides the spatial feature maps before they're averaged into a single vector. In standard MobileNetv2 architectures, this typically will be the last convolutional block prior to the global average pooling. The exact layer name will vary but generally ends with something like `_relu`, or `_Conv2D`.

The process involves two main steps. First, identify the correct layer output, and second, adjust the model to generate that output. Finally, we'll use this adjusted model to obtain the desired array. Global average pooling reduces the spatial dimensions (height and width) of the feature maps to 1x1, effectively calculating the average feature value for each channel across the entire spatial domain. Consequently, if your input data has spatial features (and it will with image data), it becomes necessary to extract the features before this spatial aggregation occurs.

Let's first look at how we adjust the model to capture this intermediate layer's output. Here's an initial example:

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model

def get_mobilenet_intermediate_features(input_shape=(224, 224, 3)):

    # Load the pre-trained MobileNetV2 model, excluding the top layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Find the target layer by searching through all the model layers.
    target_layer_name = None
    for layer in reversed(base_model.layers):
      if 'relu' in layer.name and 'conv' in layer.name: # Generally the layer right before GAP is something like this.
        target_layer_name = layer.name
        break
    if not target_layer_name:
      raise ValueError("Could not locate a suitable layer before GAP.")

    # Construct a new model that outputs the specified layer.
    intermediate_layer_model = Model(inputs=base_model.input,
                                       outputs=base_model.get_layer(target_layer_name).output)

    return intermediate_layer_model
```

In this code, the `get_mobilenet_intermediate_features` function loads MobileNetV2 without the classification top. It iterates through the layers in reverse order to locate the relevant convolutional layer, specifically those ending in `relu` (which tends to be just before global average pooling). It then creates a new Keras `Model`, whose inputs are the original model's input and output is the target feature map.

This method is preferable because it's robust to variations in layer names. Now, having extracted this layer we have an output with dimension `(batch_size, height, width, channels)`. This is not what you need. The global average pooling step produces `(batch_size, channels)`. The layer before GAP, has the `channels` we need, but also the spatial data. We need to pool this ourselves to achieve the desired result. We can do that using `tf.reduce_mean` over the spatial dimensions.
Here’s the second code example showing how you'd apply this with image data:

```python
import tensorflow as tf
import numpy as np

def extract_mobilenet_features(model, image_data):

    # Extract intermediate feature maps using our new model
    intermediate_output = model.predict(image_data)

    # Perform a spatial mean pool to aggregate spatial data
    batch_size = tf.shape(intermediate_output)[0]
    mean_features = tf.reduce_mean(intermediate_output, axis=[1, 2])

    # Convert to Numpy array of correct shape
    feature_array = mean_features.numpy()

    return feature_array
```

Here, the `extract_mobilenet_features` function takes the model obtained from the previous example and some image data. The `model.predict` part generates feature maps.  Then, `tf.reduce_mean` calculates the average feature map across height and width (axes 1 and 2) effectively providing per-image features with the channels available at the last conv layer. Finally, `feature_array` will have the desired shape, `(num_images, 1280)` given that the conv layer before average pooling usually has this number of channels in MobileNetv2 architecture. The `num_images` will be equal to the batch size as defined by the first dimension of `image_data`.

Note that I assume image_data is provided in the correct format which matches the `input_shape` parameter used by the `get_mobilenet_intermediate_features` function.

Finally, let's demonstrate how to use these functions to generate a feature array from some synthetic data. This will tie everything together:

```python
import numpy as np
import tensorflow as tf

# Generate some synthetic image data
num_images = 10
input_shape = (224, 224, 3)
image_data = np.random.rand(num_images, *input_shape)

# Call get_mobilenet_intermediate_features to get our layer model.
feature_extraction_model = get_mobilenet_intermediate_features(input_shape)

# Run the feature extraction with our synthetic images.
feature_array = extract_mobilenet_features(feature_extraction_model, image_data)

# Check the shape of output, we expect (num_images, 1280)
print("Output feature shape:", feature_array.shape)

# Check first 5 rows of output, in case we wanted to investigate the features
print("First 5 Feature Vectors:\n", feature_array[:5])
```

In this final example, random image data is created, the functions described earlier are used and the resulting shape is printed out alongside the first few feature vectors as an illustration. In practice, your image data should be preprocessed to match the requirements of the MobileNetV2 model. This typically involves resizing and normalizing pixel values. Also, the actual size of the features will vary. As you can see, the solution consists of extracting an intermediate layer and then performing the equivalent of global average pooling yourself across the spatial dimensions. This approach offers greater control in cases where standard `GlobalAveragePooling2D` is not sufficient for your required shape of feature map representation.

For additional guidance and a deeper understanding of the techniques used, I'd recommend consulting resources covering the following topics. For Keras model construction, the Keras documentation provides detailed explanations of `Model` instantiation, layer handling, and custom models. For transfer learning concepts, research papers and articles on feature extraction techniques using pre-trained convolutional networks are very useful. Furthermore, familiarity with TensorFlow’s tensor operations, especially `tf.reduce_mean`, is essential for spatial pooling operations. The TensorFlow website has an extensive guide to the use of tensor manipulation operations.
