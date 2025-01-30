---
title: "How to fix shape mismatch errors when transferring pretrained Keras model weights?"
date: "2025-01-30"
id: "how-to-fix-shape-mismatch-errors-when-transferring"
---
Pretrained Keras models, especially those from `tf.keras.applications`, are frequently employed as a starting point for various computer vision tasks. However, direct weight transfer can lead to shape mismatch errors if the target model's architecture deviates even slightly from the original, resulting in the dreaded "ValueError: Shape mismatch" during weight assignment. This problem arises because the tensors representing the weights of each layer must possess precisely matching dimensions between the pretrained and the target model for a successful transfer. I have personally encountered this issue multiple times, particularly when customizing feature extraction layers or implementing fine-tuning strategies. The core of the solution lies in understanding how Keras models structure their layers and weights, enabling targeted modifications and adjustments.

The primary source of this mismatch stems from differences in layer definitions between the source (pretrained) and target models. For example, a pretrained ResNet50 model may have a final pooling layer followed by a dense layer meant for ImageNet classification. If your target task requires a different number of classes, or if you wish to extract features prior to the dense layers, you will necessarily introduce architectural changes. Simply loading the original weights will then result in a shape mismatch because the layers do not align one-to-one. Another common reason includes variations in input shape. If the input shape of your target model is different from that of the source model, the earlier convolutional layers will eventually produce feature maps of different spatial dimensions, impacting subsequent layer compatibility. This inconsistency in feature map shapes then propagates through the network resulting in incompatibility even for seemingly similar layers. We need to manipulate the weight loading process in a way that reflects the changes we’ve made to the model's architecture.

The solution involves granular control over which weights are loaded and how, typically achieved by individually examining and loading the weights of each layer, bypassing the `model.load_weights()` method's inflexible nature. This approach involves iterating through the layers of the source and target models, and then deciding which layers to load, skipping those that have changed in the target model, such as dense layers designed for the wrong number of output classes or convolutional layers that were added or removed.

Let's consider a specific scenario where we wish to use a pretrained VGG16 for feature extraction, but only up to a specific block. We want to exclude the fully connected layers entirely as we will be attaching our own model for a different classification task. The first code example demonstrates how we would achieve this:

```python
import tensorflow as tf

# Load pretrained VGG16 model without the top layers
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define a new model using the layers from the base_model up to a specific layer.
# Note: This block number is for example, adjust this as needed
block_5_output = base_model.get_layer('block5_pool').output
feature_extraction_model = tf.keras.Model(inputs=base_model.input, outputs=block_5_output)

# Let's verify the feature extraction layers' output shapes.
# We will see output shape of (None, 7, 7, 512) before adding classification layers
print('Feature extraction model shape:', feature_extraction_model.output_shape)

# Now we define our new model, we add a flatten layer and our new custom classification layers
x = tf.keras.layers.Flatten()(feature_extraction_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x) # 10 classes example
custom_model = tf.keras.Model(inputs=feature_extraction_model.input, outputs=output)


# Now we load weights, and skip the classification layers.
for layer in feature_extraction_model.layers:
    if hasattr(layer, 'get_weights'): # ensure layer has weights
        custom_model.get_layer(layer.name).set_weights(layer.get_weights())


# Note: the custom model has the pretrained weights for blocks 1-5, but the new dense layers are randomly initialized
print('Custom model shape:', custom_model.output_shape)
```

This example illustrates the strategy of building the feature extraction model independently by selecting layers from the pretrained `base_model`, and the explicit assignment of weights layer by layer to the `custom_model`. We retrieve weights using the `get_weights()` method of each layer in the feature extraction model and set them to the correspondingly named layer in the custom model using the `set_weights()` method. By specifically excluding the later layers in our loading process, we avoid the error caused by the mismatch between the source model's and the custom model's architecture. `get_weights()` returns a list of tensors, where each list item is for example the layer weights and the biases, which are then used in `set_weights()` to set the values of the current model’s parameters to those of the pre-trained model, if and only if they exist in the current model, matching on name, and number of weights.

Another common scenario involves adapting a pretrained model to a different input image size. While we can resize images in preprocessing, this can sometimes have a negative impact on performance; it would be beneficial to transfer weights to a model with the new input shape. This can be done by rebuilding the model layers while keeping the input shape variable. In most cases, this problem can be addressed by only building the model from layers, and then loading the weights once the layers are instantiated. Note that models like `ResNet`s, and `VGG`s use a pooling layer after the convolutional blocks which reduce dimensionality and therefore mitigate this problem somewhat, but it is still a potential source of shape mismatches.

```python
import tensorflow as tf

# Load a pretrained model without specifying an input shape
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Define a new input shape
new_input_shape = (128, 128, 3)

# Build a new model, keeping the architecture but changing the input shape
inputs = tf.keras.Input(shape=new_input_shape)
x = inputs

for layer in base_model.layers[1:]: # We skip the first input layer of the base_model
    x = layer(x)

custom_model = tf.keras.Model(inputs=inputs, outputs=x)

# Load weights of the original layers using the correct indices. Note that this has been modified to not
# include the first layer which has been added above.
for idx, layer in enumerate(base_model.layers[1:]):
    if hasattr(layer, 'get_weights'):
        custom_model.layers[idx+1].set_weights(layer.get_weights())


# Let's verify the input and output shapes of the custom model and base model
print(f'Base model input shape: {base_model.input_shape}')
print(f'Base model output shape: {base_model.output_shape}')
print(f'Custom model input shape: {custom_model.input_shape}')
print(f'Custom model output shape: {custom_model.output_shape}')

# Add custom layers and train accordingly
x = tf.keras.layers.Flatten()(custom_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x) # 10 classes example
final_model = tf.keras.Model(inputs=custom_model.input, outputs=output)
```

Here, we rebuild the target model using the layer definitions from the pretrained model, adjusting the initial input layer and shape using `tf.keras.Input` before transferring the pre-trained weights. The weights are again transferred layer by layer, matching on names and the number of weights. This approach allows us to leverage the learned features while adapting the network to the desired input size. We explicitly iterate through layers from index `1` and then use index `idx+1` for setting the weights of the custom model, due to the input layer.

Finally, consider a scenario where a block in a pretrained model has been modified. For example, we might want to change the number of filters in a specific convolution layer to better suit our dataset.

```python
import tensorflow as tf
# Load a pretrained VGG16 model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = inputs

for layer in base_model.layers:
    if layer.name == 'block3_conv2':
        # Replace block 3 conv2 layer with a modified one
        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', name='block3_conv2_modified')(x)
    else:
        x = layer(x)

custom_model = tf.keras.Model(inputs=inputs, outputs=x)

# Load weights for all layers, skipping the one we modified
for layer in base_model.layers:
  if hasattr(layer, 'get_weights'):
    if layer.name != 'block3_conv2':
       custom_model.get_layer(layer.name).set_weights(layer.get_weights())


#Check model
print(f'Base model output shape {base_model.output_shape}')
print(f'Custom model output shape {custom_model.output_shape}')
```
In this example, we modify one specific convolution layer, and we then explicitly exclude the layer named `block3_conv2` when we load the weights. We also ensure that the new layer's name is different from the original, avoiding mismatches in the weight loading. This allows us to selectively transfer only the relevant weights while ignoring the modified layer.

These examples demonstrate how careful iteration and layer-by-layer weight assignment provide the flexibility needed to work around shape mismatches. By inspecting the structure of your source and target models, and then using the `get_weights()` and `set_weights()` methods, shape mismatch issues during pretrained model weight transfer can be effectively addressed.

For further reference, consider consulting the official Keras documentation, specifically the sections on model building and saving/loading weights. Also, the TensorFlow guides offer detailed explanations of model layers and construction. Books such as "Deep Learning with Python" by Chollet can provide additional clarity about model architecture and weight manipulation. Additionally, practical tutorials can be found on the TensorFlow website. These resources should further your understanding of model building and weight loading strategies in Keras, and give you more tools for avoiding and correcting the "ValueError: Shape mismatch" when working with pretrained models.
