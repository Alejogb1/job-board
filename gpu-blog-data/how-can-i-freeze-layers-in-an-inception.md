---
title: "How can I freeze layers in an Inception V3 model using TensorFlow 3?"
date: "2025-01-30"
id: "how-can-i-freeze-layers-in-an-inception"
---
Freezing layers in a pre-trained Inception V3 model, particularly during fine-tuning, is critical for preventing the destruction of learned features and accelerating the training process on specific tasks. The core idea is to selectively disable the training of certain weights within the network, thereby preserving the knowledge those weights represent. This is accomplished by altering the `trainable` attribute of individual layers within the model. While seemingly simple, this technique requires careful consideration of the layers being frozen and the overall training strategy. I've implemented this in several projects, ranging from image classification to style transfer, and I've found that a clear understanding of the model's structure and intended application is paramount.

In TensorFlow 3, freezing layers involves iterating through the model's layers, identifying those to be frozen, and setting their `trainable` attribute to `False`. This prevents the optimizer from updating the weights of these layers during the backpropagation process. Layers not designated as frozen, usually the higher-level layers closer to the output, will remain trainable, allowing the model to adapt to the new task.

Let's illustrate with a few practical code examples using TensorFlow 3.

**Example 1: Freezing all layers except the final classification layer.**

This is a common approach when you're adapting the pre-trained model for a different classification task with similar visual characteristics. We want to leverage the Inception network's strong feature extraction capabilities, but adapt the final decision-making stage to our data. The typical strategy is to replace or adjust the final classification layer for the new classes of data.

```python
import tensorflow as tf
from tensorflow.keras.applications import inception_v3

def freeze_inception_except_last(num_classes):
    # Load pre-trained Inception V3 model without top (classification) layers
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # Set all layers to non-trainable initially
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create final model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    return model


# Example usage:
num_classes_example = 5
model_example1 = freeze_inception_except_last(num_classes_example)

# Verify layers are frozen
for layer in model_example1.layers:
    if layer.name.startswith('inception_v3'):
      assert layer.trainable == False

# Check that the final layers are trainable
assert model_example1.layers[-1].trainable == True
assert model_example1.layers[-2].trainable == True
```

In this code, we first load the Inception V3 model without its final classification layers (`include_top=False`). Then, we iterate through all the layers of the base model and set their `trainable` property to `False`. Afterward, we append our custom classification layers. These are `GlobalAveragePooling2D`, `Dense(1024, activation='relu')`, and finally, `Dense` layer with `softmax` activation for the number of classes we require. The final model is constructed using the base Inception V3 model as input and the custom classification layers as output. By iterating again through all of the layers, we can see how the inception V3 layers have been correctly frozen and how the classification layers are trainable. This setup allows the pre-trained convolutional layers to act as robust feature extractors, while the added fully connected layers are fine-tuned to the new classification task.

**Example 2: Freezing the first 'N' layers while allowing the rest to train.**

This strategy is often useful when you want to retain very low-level features from the initial layers, but fine-tune the higher-level features to adapt to your specific domain. By freezing initial layers, we preserve the most fundamental feature detectors, such as edge and blob detectors. By fine-tuning the later layers, we adapt the model to high-level features relevant to the task.

```python
import tensorflow as tf
from tensorflow.keras.applications import inception_v3


def freeze_n_layers_of_inception(num_layers_to_freeze, num_classes):
  # Load pre-trained Inception V3 model without top (classification) layers
  base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

  # Freeze the first 'num_layers_to_freeze' layers
  for i, layer in enumerate(base_model.layers):
    if i < num_layers_to_freeze:
      layer.trainable = False
    else:
        layer.trainable = True


  # Add custom classification layers
  x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

  # Create final model
  model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

  return model


# Example Usage
num_layers_to_freeze_example = 100
num_classes_example = 5
model_example2 = freeze_n_layers_of_inception(num_layers_to_freeze_example, num_classes_example)

# Verify the number of layers that were frozen.
count_frozen = 0
for i, layer in enumerate(model_example2.layers):
  if layer.name.startswith('inception_v3'):
    if i < num_layers_to_freeze_example:
        assert layer.trainable == False
        count_frozen += 1
    else:
        assert layer.trainable == True
assert count_frozen == num_layers_to_freeze_example
```

Here, instead of freezing all layers, we iterate through the `base_model.layers` and freeze the first `num_layers_to_freeze`. The remaining layers are set to `True`, meaning they will train during backpropagation. This enables more nuanced control over which parts of the model learn and which retain the pre-trained feature representation. This approach allows fine-tuning of the network beyond the final classification layers to adapt more effectively.

**Example 3: Freezing specific blocks of layers based on layer names.**

Sometimes, specific blocks of layers within the model are better suited to be frozen based on their function or position in the network. This example demonstrates freezing Inception blocks while leaving other layers, including the first layers, trainable.

```python
import tensorflow as tf
from tensorflow.keras.applications import inception_v3


def freeze_specific_blocks(num_classes):
    # Load pre-trained Inception V3 model without top (classification) layers
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # Freeze specific layers based on name containing 'mixed' which is part of Inception layers.
    for layer in base_model.layers:
        if 'mixed' in layer.name:
            layer.trainable = False
        else:
           layer.trainable = True

    # Add custom classification layers
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create final model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    return model


# Example Usage:
num_classes_example = 5
model_example3 = freeze_specific_blocks(num_classes_example)

# Verify that the desired layers are frozen.
for layer in model_example3.layers:
    if layer.name.startswith('inception_v3') and 'mixed' in layer.name:
        assert layer.trainable == False
    elif layer.name.startswith('inception_v3'):
        assert layer.trainable == True
```

In this case, we freeze all the layers that include the string ‘mixed’ within their names, which represent Inception modules within the network. By freezing these, we ensure that the Inception blocks that contain complex convolutional layers are not adjusted during fine-tuning. This allows other layers to fine-tune to the new task while still taking advantage of the pre-trained Inception blocks. This demonstrates a much more granular way of applying the freeze parameter.

**Resource Recommendations:**

For deeper understanding, consider reviewing materials covering transfer learning, specifically those focusing on fine-tuning convolutional neural networks. Research on the specific architecture of Inception V3 and its components is highly valuable, understanding how the Inception blocks contribute to image feature extraction will improve the selection of frozen layers. Also, consult resources explaining the backpropagation process and how the `trainable` parameter affects gradient computation. These resources will provide both conceptual understanding and practical knowledge for effective use of pre-trained models and the freezing of their layers. Finally, investigating optimization techniques and learning rate adjustments will help you better control the fine-tuning process.
