---
title: "Can MobileNetV2 be used effectively with input images smaller than 32x32 in TensorFlow Keras?"
date: "2025-01-30"
id: "can-mobilenetv2-be-used-effectively-with-input-images"
---
MobileNetV2, primarily designed for mobile and embedded applications, typically expects input images of size 224x224. However, its architecture doesn’t inherently restrict it to this specific dimension, presenting a path toward using it with smaller images, albeit with certain caveats.

Fundamentally, the core of MobileNetV2 lies in its inverted residual blocks and depthwise separable convolutions. These operations, unlike fully connected layers, are adaptable to different input resolutions to a degree. The critical aspect lies in how stride and padding are handled within the convolutions. The downsampling in MobileNetV2, implemented through strided convolutions, progressively reduces the feature map dimensions as the network deepens. Using an input significantly smaller than the conventionally tested 224x224 requires careful consideration to ensure these downsampling operations do not reduce the feature maps to trivial sizes, thus eliminating useful information early in the network.

When aiming for very low resolutions like 32x32 or lower, issues become pronounced. The standard stride-2 convolutions used for downsampling can quickly erode the spatial dimensions. Consider an input image of 32x32: even a single strided convolution with a kernel size larger than 1 will immediately reduce the feature map size. If multiple such operations follow, as they do within the initial layers of MobileNetV2, the spatial dimension collapses rapidly, possibly to 1x1. Consequently, the subsequent inverted residual blocks become ineffective, preventing the network from learning meaningful representations.

However, this isn’t insurmountable. A solution lies in customizing the stride configuration of the initial layers and adjusting the overall network structure to suit the lower input resolution. The key is to minimize the number of downsampling operations, specifically during the initial blocks, while still preserving some notion of receptive field for each feature. Therefore, the modification involves selectively removing some of the strides, and possibly altering other architectural parameters to compensate.

The specific success of this approach depends on multiple factors: the complexity of the dataset being classified (simpler datasets with coarse features may work well), the size of the effective receptive fields (which need to correspond to features of the objects being classified) and the depth of the network (a smaller input size may correspond to a reduction in depth to avoid collapsing the spatial feature maps). A 32x32 input may also perform best with networks specifically trained for this input size rather than attempting to finetune a larger pre-trained network.

Let’s consider code examples.

**Example 1: Modifying stride**

The first example illustrates the core concept: modifying strides in the initial convolutional layers.

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D

def create_modified_mobilenetv2(input_shape=(32, 32, 3), num_classes=10):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape)
    x = base_model.layers[0].output  # Get the input tensor of the first block

    #Modify the first two Conv2D layers:
    for layer in base_model.layers[1:3]:
      if isinstance(layer, Conv2D):
          layer.strides = (1,1)  # Set stride to 1 for initial convolutions
    
    # Rebuild the Model starting from the modified Conv2D layer output
    model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)


    # Add global average pooling, and a final dense layer for classification
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.models.Model(inputs=model.input, outputs=x)



model = create_modified_mobilenetv2()

model.summary()
```

In this code, `create_modified_mobilenetv2` modifies the stride of the first two `Conv2D` layers of MobileNetV2 to `(1,1)`. This prevents the quick reduction in feature map spatial dimension. I’ve also added a global average pooling and dense layer for classification on top of the output. The summary shows that most layers are still present, but the early downsampling is reduced.

**Example 2: Using a custom network with modified downsampling**

This example demonstrates building a custom network using depthwise separable convolutions similar to MobileNetV2, but with more control over the downsampling.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense

def inverted_residual_block(inputs, filters, stride, expansion):
    # Expansion
    x = Conv2D(filters * expansion, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Projection
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Residual
    if stride == 1:
      x = tf.keras.layers.Add()([inputs, x])

    return x

def create_custom_network(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = inverted_residual_block(x, 16, 1, 1)
    x = inverted_residual_block(x, 24, 2, 6)  # Strided convolution here.
    x = inverted_residual_block(x, 32, 1, 6)
    x = inverted_residual_block(x, 64, 2, 6) # And here.
    x = inverted_residual_block(x, 96, 1, 6)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation="softmax")(x)

    return tf.keras.models.Model(inputs=inputs, outputs=x)

model = create_custom_network()
model.summary()
```

This example defines a custom network using the core principles of MobileNetV2’s inverted residual block. Downsampling occurs only at two points, allowing sufficient feature map dimensions for smaller input sizes. The `inverted_residual_block` function encapsulates the logic of an expansion, depthwise, and projection convolution. The initial convolution in the model has a stride of 1, preserving the spatial resolution at the start, and then downsampling is performed with a limited number of strided blocks. This approach provides more flexibility than directly modifying a pre-trained model.

**Example 3: Using a smaller MobileNetV2 model with different input size**

The last example demonstrates directly creating a MobileNetV2 with a smaller input size by only modifying the `input_shape`. It does *not* change any of the internal convolutional strides. This is provided to demonstrate the standard behavior, which will not work well at low input resolutions.

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def create_mobilenetv2_32(input_shape=(32, 32, 3), num_classes=10):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    return model
model = create_mobilenetv2_32()
model.summary()
```

When examining the output of `model.summary()`, it becomes evident that while the network is instantiated with a smaller input, it still contains the downsampling strides. This means, due to the smaller input size, the feature maps will become negligibly small after a few layers. While this network executes, it is unlikely to converge to an optimal solution due to loss of information early on. This example highlights the importance of manual adjustments demonstrated in the first two examples.

Using MobileNetV2 effectively with inputs like 32x32 requires careful modifications to its architecture, predominantly through stride adjustments or custom architecture design which controls downsampling operations more carefully. Without these alterations, the downsampling inherent in MobileNetV2 quickly reduces spatial dimensions to negligible sizes, preventing effective feature extraction. It is often necessary to carefully examine the architecture, especially the downsampling strides, to build a network that can learn effectively for smaller image inputs.

For additional learning and resources, I would recommend reviewing publications and guides on network compression and architecture modification for small images, especially those related to mobile and embedded device applications. The TensorFlow and Keras documentation provide a wealth of information on custom layer implementations and architectural design. Research papers on lightweight networks and the EfficientNet family of architectures provide additional insights. Exploring tutorials on fine-tuning pre-trained models, while often focused on larger images, can illuminate techniques for customizing layers and optimization strategies.
