---
title: "How can I freeze certain filters in a pre-trained TensorFlow model?"
date: "2025-01-30"
id: "how-can-i-freeze-certain-filters-in-a"
---
The challenge of selectively freezing layers in a pre-trained TensorFlow model, particularly convolutional filters, stems from the imperative to retain beneficial features learned during prior training while adapting the model for a new, often related, task. This is a cornerstone of transfer learning, a technique I’ve utilized extensively in my work on medical image analysis, where vast, labeled datasets are scarce. Specifically, instead of retraining a model from scratch, we often leverage weights from a network pre-trained on a large corpus like ImageNet and then fine-tune it for the medical domain. Freezing allows us to prevent the re-initialization and update of filters in specific layers.

The fundamental mechanism in TensorFlow for controlling which parameters are trainable resides in each layer's `trainable` attribute. By default, all layers in a model are trainable. To freeze specific layers, you need to iterate through the model's layers and set this attribute to `False` for the desired layers. This will prevent the model's optimizer from computing gradients for those layers, effectively holding their weights constant during the training process. It's important to recognize that the `trainable` flag impacts all parameters associated with a given layer (e.g., convolutional weights, biases). One cannot directly freeze individual filters within a single convolutional layer via this approach. To freeze a specific subset of *weights*, not entire *layers*, more sophisticated techniques are required and not addressed by the context of this question.

Let's illustrate with some concrete examples, each focusing on a slightly different scenario.

**Example 1: Freezing the Entire Base Model**

The most basic case involves freezing all layers of a pre-trained model, typically a feature extractor. This would commonly be done when appending new layers, such as dense layers, to perform classification or regression on top of the pre-trained features. This method is helpful when the target domain dataset is relatively small, allowing more rapid training of the added layers without losing the benefit of the pre-trained feature space.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Load a pre-trained VGG16 model, excluding the classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new layers for a different task
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x) # Example: 10 class classification

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Verify if model is actually frozen.
for layer in model.layers[:len(base_model.layers)]:
  print(layer.name, layer.trainable)

# Compile and train as needed.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

In this example, I load the VGG16 model, excluding its classification head, then iterate through each layer, setting `trainable` to `False`. This prevents modifications to the pre-trained weights during training. Finally, I add new layers to adapt the model for the target task, compile it, and proceed with training. Note the print statements added to demonstrate layer names and whether or not they are trainable. This is crucial for debugging and sanity checking.

**Example 2: Freezing Specific Layers by Name**

It is often desirable to freeze only the early layers while fine-tuning later ones. This allows fine-grained control over the model training. For instance, lower convolutional layers often capture basic features like edges and corners, which can be beneficial across many tasks. Freezing these layers while training later layers that capture more task-specific features can lead to better results.

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load a pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Layers to freeze by name.
freeze_layers = ['conv1_conv', 'bn_conv1', 'conv2_block1_1_conv','conv2_block1_2_conv']

# Freeze specific layers by name
for layer in base_model.layers:
  if layer.name in freeze_layers:
    layer.trainable = False
  else:
    layer.trainable = True

# Add new layers for a different task
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x) # Example: Binary classification

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Verify if model is actually frozen.
for layer in model.layers[:len(base_model.layers)]:
  print(layer.name, layer.trainable)

# Compile and train.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

```
Here I load a ResNet50 model. Instead of freezing all base model layers, I define a list, `freeze_layers` which contain string names of layers. During the iteration through the base model layers, if a layer's name is in the `freeze_layers` list, it's trainable property is set to `False`. Other layers are set to trainable. This mechanism allows me to choose which layers are involved during gradient computation during training. The rest of the code remains the same as before.

**Example 3: Freezing Layers based on Index**

Sometimes, you might not know the layer names beforehand or want to freeze based on their position within the model. The ability to freeze layers using indices can prove to be useful in such a scenario. This example freezes the initial 10 layers using index-based selection within the same ResNet50 Model.

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load a pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the first n layers
num_freeze_layers = 10
for i in range(len(base_model.layers)):
    if i < num_freeze_layers:
        base_model.layers[i].trainable = False
    else:
        base_model.layers[i].trainable = True

# Add new layers for a different task
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x) # Example: Multi-class classification

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Verify if model is actually frozen.
for layer in model.layers[:len(base_model.layers)]:
  print(layer.name, layer.trainable)

# Compile and train.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
Here I load the same ResNet50 model. Instead of freezing by name, I iterate through the layers by index, using the first 10 indexes as the subset of layers to freeze, by setting the `trainable` property to `False`. The rest of the model behaves the same as the previous examples.

**Resource Recommendations**

To further develop your understanding of transfer learning and fine-tuning in TensorFlow, I recommend focusing on the following resources. The official TensorFlow documentation is an invaluable reference point for API specific details and best practices. The *Deep Learning with Python* book by François Chollet is very practical and provides hands-on guidance on the concepts discussed. The *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* book by Aurélien Géron also provides concrete examples in machine learning and has dedicated sections for transfer learning. Finally, look into published academic research papers focused on transfer learning, which can often be found through sites like Google Scholar. These will contain the scientific basis for these strategies. By cross-referencing these materials, you will develop a robust understanding for selectively freezing layers in pre-trained TensorFlow models. These resources go a step further in outlining how to choose a correct strategy, an important task.
