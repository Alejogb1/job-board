---
title: "How can CNN ResNet 50 be fine-tuned?"
date: "2024-12-23"
id: "how-can-cnn-resnet-50-be-fine-tuned"
---

Alright, let's tackle fine-tuning a ResNet50. I’ve spent a fair bit of time working with convolutional neural networks, including plenty of iterations on ResNet architectures. It’s a powerful backbone, but like any tool, it performs best when properly configured for a specific task. The vanilla, ImageNet-trained version, is excellent for general feature extraction, but seldom provides peak performance out-of-the-box for your custom dataset. Fine-tuning is the necessary bridge to cross that gap.

First, let’s be clear on what fine-tuning *actually* entails. We're essentially taking a pre-trained model – in this case, a ResNet50 trained on a massive dataset like ImageNet – and adapting its learned parameters to a new, and usually smaller, dataset relevant to our target task. We're leveraging transfer learning – the idea that features learned on one problem can be beneficial for another, related problem. The key is to avoid training the network from scratch, which would be computationally expensive and require massive amounts of labeled data. We're starting with an excellent set of filters and leveraging that advantage.

The core concept hinges on updating, or *fine-tuning*, the weights of the network. There are a few strategic approaches. One method is to freeze the initial layers and only update those towards the end, closer to the classification output. Given that early layers often capture low-level features (edges, corners), these are less likely to need substantial adjustment. The later layers, conversely, tend to learn higher-level features specific to the ImageNet dataset; those need adaptation for the specifics of your new images and labels. Another, more aggressive strategy is to unfreeze more of the network or even all of it and train on the new data with a smaller learning rate. This can be effective but carries the risk of catastrophically forgetting previous knowledge, also called catastrophic forgetting. The best path is almost always empirical; experimentation and careful monitoring are crucial.

Now, let’s make that concrete with some code examples. I'll use Python with TensorFlow/Keras, since it's the environment I often find myself in.

**Example 1: Freezing Initial Layers**

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load pre-trained ResNet50 model, excluding the top (fully connected) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers of the base model
for layer in base_model.layers:
  layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x) # 'num_classes' is the number of classes for your task.

# Combine model
model = Model(inputs=base_model.input, outputs=predictions)

# Define optimizer and compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=validation_dataset) # assumes 'train_dataset' and 'validation_dataset' are appropriately preprocessed tf.data.Dataset objects
```

In this example, we've loaded a pre-trained ResNet50, stripped off the classification head, and replaced it with our custom layers. All convolutional layers are frozen. The training process now only adjusts the newly added dense layers. This approach is safe and a good starting point for most fine-tuning scenarios with a dataset of moderate size.

**Example 2: Unfreezing later layers with reduced learning rate**

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load pre-trained ResNet50 model, excluding the top (fully connected) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze layers for fine-tuning, but only the later ones
fine_tune_at = len(base_model.layers) // 2 # we unfreeze only the final half of layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combine model
model = Model(inputs=base_model.input, outputs=predictions)


# Define optimizer and compile the model with reduced learning rate
optimizer = Adam(learning_rate=0.00001) # lower learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
```
Here, we have a similar structure, but now we have unfrozen the second half of the base model's layers to allow more flexibility. Crucially, we've reduced the learning rate of the Adam optimizer significantly. This ensures that we avoid rapid weight changes in the unfrozen layers and maintain some of the pre-trained knowledge while allowing the unfrozen layers to adapt to the target data.

**Example 3: Differential Learning Rates**

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load pre-trained ResNet50 model, excluding the top (fully connected) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define a dictionary to specify learning rates per layer (or group of layers)
layer_learning_rates = {}
for i, layer in enumerate(base_model.layers):
    if i < len(base_model.layers) // 4:
        layer_learning_rates[layer.name] = 0.000001  # very low rate for initial layers
    elif i < len(base_model.layers) * 3 // 4:
        layer_learning_rates[layer.name] = 0.00001   # low rate for mid layers
    else:
        layer_learning_rates[layer.name] = 0.0001    # regular rate for final layers

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combine model
model = Model(inputs=base_model.input, outputs=predictions)

# Custom optimizer that applies different learning rates to different layers
class LayerSpecificAdam(Adam):
    def __init__(self, layer_learning_rates, **kwargs):
        super().__init__(**kwargs)
        self.layer_learning_rates = layer_learning_rates

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
      updated_grads_and_vars = []
      for g, v in grads_and_vars:
        if v.name.split('/')[0] in self.layer_learning_rates:
          new_g = g * self.layer_learning_rates[v.name.split('/')[0]]/ self.lr.value()
        else:
          new_g = g
        updated_grads_and_vars.append((new_g, v))
      super().apply_gradients(updated_grads_and_vars, *args, **kwargs)

# Use custom optimizer
optimizer = LayerSpecificAdam(layer_learning_rates = layer_learning_rates, learning_rate = 0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
```

This example introduces a sophisticated approach: differential learning rates. The `LayerSpecificAdam` optimizer customizes the learning rate per layer. This enables us to carefully tune even more precisely and aggressively. Low-level layers will train slowly while higher level ones will learn more quickly. This often yields better accuracy and avoids the instability of unfreezing the whole network with the same high rate.

To delve deeper into this subject, I'd recommend exploring the original ResNet paper, "Deep Residual Learning for Image Recognition" by He et al. (2015) to truly understand the architecture. "Deep Learning" by Goodfellow, Bengio, and Courville provides a very detailed theoretical background in general. Also, looking into the *Transfer Learning* chapter within "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Géron, can be really helpful. Experimentation and a good understanding of your dataset is paramount when fine-tuning. Never underestimate the value of systematically assessing the impact of each architectural change or training parameter.
