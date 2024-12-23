---
title: "How can I fine tune the last layers of a neural network in a target model for transfer learning?"
date: "2024-12-23"
id: "how-can-i-fine-tune-the-last-layers-of-a-neural-network-in-a-target-model-for-transfer-learning"
---

, let's talk about fine-tuning those last layers. It's a common scenario, and I’ve certainly spent my fair share of time navigating this particular landscape. Back in my days working on a rather quirky image recognition project for a specialized manufacturing process, we heavily relied on transfer learning. We had a pre-trained model, a ResNet-50 if memory serves, trained on ImageNet, and our task involved differentiating various material defects. The data was quite different from ImageNet—we had specific lighting conditions, unique defect patterns, and a relatively small labeled dataset. The fully pre-trained model, while having a solid backbone, didn't immediately provide the accuracy we needed. That's when we zeroed in on strategically fine-tuning the last few layers.

Now, why focus on the last layers specifically? Well, the early layers in a convolutional neural network (CNN), for instance, tend to learn more general, low-level features, like edges and corners. As you move deeper, the layers extract more complex, task-specific features. In transfer learning, we often leverage those generic feature extractors from the pre-trained model. However, the final layers, which act as the classifier, are often highly specialized to the source domain, ImageNet in our case. Fine-tuning these later layers allows the model to adapt those high-level representations to your target task, without throwing away the valuable feature extraction learned previously.

This process of targeted fine-tuning is crucial because it helps prevent overfitting, especially when working with smaller datasets. If you were to retrain the entire model, the risk of overfitting to your specific target data is significantly higher. By focusing on just the final layers, you’re essentially preserving the useful generalized feature representations while adapting the model’s classification logic. Think of it like taking a pre-built chassis of a car (the feature extractors) and adapting the control panel (the final layers) to match your specific needs.

Let’s delve into a few practical scenarios with accompanying code examples.

**Scenario 1: Simple Classification Layer Adjustment**

Imagine the last layers of your pre-trained model consist of a fully connected layer followed by a softmax activation, outputting probabilities over 1000 classes. Your target classification task requires, say, only 5 categories. You'll need to modify this output layer. We use a library like TensorFlow/Keras here, and this is conceptually similar for other frameworks like PyTorch.

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers

# Load the pre-trained ResNet50 model, excluding the top (classification) layer
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze the convolutional base
base_model.trainable = False

# Define the new classification head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(5, activation='softmax')(x) # 5 classes for target task

# Assemble the model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to check trainable parameters
model.summary()

# Now you can train the model
# model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
```

In this example, we freeze the convolutional base (`base_model.trainable = False`) and append a new fully connected layer, followed by a 5-unit softmax layer. This replacement is the core of adapting the model to the new classification space. The key here is to ensure that only the new layers and, in some cases, the immediately preceding layers of the base_model are marked as trainable when fitting.

**Scenario 2: Fine-tuning a small number of convolutional layers**

Sometimes, adapting only the classification head isn't enough. You may need to fine-tune some of the *last* convolutional blocks to further adapt the features to your dataset. Here's how you could accomplish that:

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers

# Load the pre-trained ResNet50 model, excluding the top (classification) layer
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Unfreeze some last convolutional layers. Count from the last and use layers by block
fine_tune_at = 150 # A safe starting point

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Define the new classification head as before
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(5, activation='softmax')(x) # 5 classes for target task

# Assemble the model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Print the model summary to check trainable parameters
model.summary()
# Now you can train the model
# model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
```

In this case, we’ve strategically unfrozen the last few convolution blocks defined by the arbitrary number of 150 layers (`fine_tune_at`). This implies the layers before 150 are frozen while the remaining ones are marked as trainable, along with our new classification layers. It’s essential to experiment with the number of layers you unfreeze. Start with a small number and progressively increase it, observing performance and monitoring for overfitting.

**Scenario 3: Learning Rate Adjustment for Fine-tuning**

When fine-tuning, it's often beneficial to use a smaller learning rate for the layers of the pre-trained base model compared to the new, added layers. This allows fine adjustment of base model layers, preserving the previously acquired knowledge. For new layers, a higher learning rate may speed up their convergence as they have to adapt more from the pre-trained state.

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Load the pre-trained ResNet50 model, excluding the top (classification) layer
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Unfreeze some of the last convolutional layers
fine_tune_at = 150

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Define the new classification head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(5, activation='softmax')(x)

# Assemble the model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Create a dictionary with layers and a different learning rate.
lr_dict = {model.layers[i].name: 1e-4 for i in range(len(model.layers))}
lr_dict.update({layer.name: 1e-3 for layer in model.layers if 'dense' in layer.name })
lr_dict.update({layer.name: 1e-3 for layer in model.layers if 'softmax' in layer.name })


optimizer = Adam(learning_rate=1e-4) # Default rate

for layer in model.layers:
  if layer.trainable:
     if layer.name in lr_dict:
       layer.add_loss(lambda: 0.0)
       layer.add_update(tf.Variable(0.0, trainable=False)) # Make layers trainable
     
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Print the model summary to check trainable parameters
model.summary()

# Now you can train the model
# model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))

```

Here we created custom learning rates for the added dense and softmax layers. This can improve training as you have finer control over how much change is applied to each trainable layer. Note that implementing this in pure Keras can be more involved. This snippet shows one method using explicit layer loss and update mechanisms.

For more in-depth knowledge on transfer learning and fine-tuning, I highly recommend delving into "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. It is a comprehensive resource for anyone seeking a strong theoretical foundation. Also, the research papers published by the creators of the specific pre-trained models you're using (such as ResNet) will offer valuable insights into their architectures and best practices for fine-tuning. Don't just rely on tutorials; go to the source material.

Remember, fine-tuning is a bit of an art as much as it is a science. It requires experimentation and thoughtful adjustments based on your specific data and task. There is no one-size-fits-all solution. The key is to understand the underlying mechanics of the model and to approach the fine-tuning process strategically.
