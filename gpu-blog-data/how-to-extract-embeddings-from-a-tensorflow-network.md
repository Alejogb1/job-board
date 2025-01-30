---
title: "How to extract embeddings from a TensorFlow network?"
date: "2025-01-30"
id: "how-to-extract-embeddings-from-a-tensorflow-network"
---
The ability to extract intermediate layer activations, often referred to as embeddings, is fundamental for tasks such as transfer learning, feature visualization, and similarity analysis in deep learning. In TensorFlow, this process involves manipulating the computational graph to selectively expose the desired layer outputs. I’ve routinely implemented this in various projects, from custom image similarity search engines to hybrid recommendation systems, and the techniques always revolve around accessing the network's architecture.

The core principle is to define a model that, instead of producing the final prediction (e.g., classification probabilities), returns the output of the target embedding layer. This bypasses the classification layers and directly provides the learned representation. The necessary steps involve accessing the model’s intermediate layers and constructing a new model that targets these layers. I typically use one of two approaches, depending on the situation: either I modify an existing model or create a separate, smaller model based on the original. Modifying the existing model is less common for me as it might alter the training paradigm. I favor creating a dedicated model, as this isolates the embedding extraction functionality.

The first, and most common, step is to load your pretrained model. This might be a model saved using `tf.keras.models.save_model` or a model loaded from TensorFlow Hub. After successfully loading the model, identifying the target embedding layer is next. This involves knowing the name or layer object of the layer you wish to extract features from. I usually rely on the Keras model’s `summary()` method or a similar inspection tool to determine the names and layer types, or my own notes taken during model creation. These layers are commonly dense layers, convolutional layers after flattening, or specific attention layers.

Once the target layer is known, a new model is created with the same input as the original model, but the output is set to the output of the desired layer. This is done using `tf.keras.Model`, specifying the inputs and outputs. This new model essentially acts as a forward pass for your original model, but only up to the selected layer. It is important to set the training parameter for this new model to `False` in most scenarios, especially if the original model was trained, as the embedding model should be used for feature extraction and not training.

Here are a few code examples that demonstrates this process:

```python
import tensorflow as tf
import numpy as np

# Example 1: Embedding Extraction from a Custom Sequential Model

# Assume a pre-trained model for image classification
def create_example_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', name='embedding_layer'), # Target Embedding layer
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

pretrained_model = create_example_model() # Usually loaded from file

# The target embedding layer is accessed by name
embedding_layer = pretrained_model.get_layer('embedding_layer')

# The embedding model is constructed by specifying the inputs and outputs
embedding_model = tf.keras.Model(inputs=pretrained_model.input, outputs=embedding_layer.output)
embedding_model.trainable = False # Prevents accidental updates

# Dummy input for demonstration
dummy_input = np.random.rand(1, 28, 28, 1)

# Get the embedding
embeddings = embedding_model.predict(dummy_input)
print("Embedding shape:", embeddings.shape)
```
In this first example, a custom model is created for demonstration. The key part is targeting the dense layer named `embedding_layer`. I then create `embedding_model` by passing the original model's inputs and the outputs of the identified layer.  The `embedding_model` is then used to generate the output embeddings given a dummy input. The `trainable = False` prevents inadvertent weight modifications in the embedding model. This is the standard approach for extracting the output of the layers in sequential models.
```python
import tensorflow as tf
import numpy as np

# Example 2: Embedding Extraction from a Functional API Model
def create_functional_model():
    input_tensor = tf.keras.layers.Input(shape=(32,))
    x = tf.keras.layers.Dense(64, activation='relu')(input_tensor)
    embedding_output = tf.keras.layers.Dense(128, activation='relu', name='embedding_layer')(x) # Target Embedding layer
    output_tensor = tf.keras.layers.Dense(10, activation='softmax')(embedding_output)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model

pretrained_model_functional = create_functional_model()

# Target embedding layer can be accessed by its name
embedding_layer = pretrained_model_functional.get_layer('embedding_layer')


# Construct the embedding model using inputs and target outputs
embedding_model_functional = tf.keras.Model(inputs=pretrained_model_functional.input, outputs=embedding_layer.output)
embedding_model_functional.trainable = False

# Dummy input for demonstration
dummy_input_functional = np.random.rand(1, 32)

# Extract embeddings
embeddings_functional = embedding_model_functional.predict(dummy_input_functional)
print("Embedding shape from functional model:", embeddings_functional.shape)
```
In the second example, I demonstrate the embedding extraction process on a model built using the functional API. It’s almost identical to the sequential model approach, with the key difference being the way I access the embedding layer within a more complex graph. The logic remains the same: identifying the desired layer by name, constructing a new model with targeted outputs, and using that model for prediction. The usage of `trainable = False` is still critical.

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16


# Example 3: Embedding Extraction from a Pre-trained Model (VGG16)
pretrained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Access the last block of convolution layer in the VGG16
embedding_layer = pretrained_vgg.get_layer('block5_conv3')


# Create the embedding model
embedding_vgg = tf.keras.Model(inputs=pretrained_vgg.input, outputs=embedding_layer.output)
embedding_vgg.trainable = False

# Dummy input for VGG16
dummy_input_vgg = np.random.rand(1, 224, 224, 3)

# Get embeddings
embeddings_vgg = embedding_vgg.predict(dummy_input_vgg)
print("Embedding shape from VGG:", embeddings_vgg.shape)
```

The final example tackles the embedding extraction from a pre-trained VGG16 model, which is commonly used in image-related feature extraction tasks. I load the model with `include_top=False`, indicating that the final classification layers are not needed. Then, I target the output of the 'block5_conv3' layer. As before, a new model is created that outputs the activation of this targeted layer. This approach can be generalized to any model available in `tf.keras.applications` or even custom pre-trained models.

In practice, post-processing may be needed on the extracted embeddings.  For example, if the extracted output is a tensor from a convolutional layer, it might be necessary to average pool this to obtain a fixed-size embedding. Such operations depend upon the specific problem. It is also often beneficial to perform normalization or dimensionality reduction using techniques like Principal Component Analysis.

For resource recommendations, I suggest delving deeper into the TensorFlow API documentation, specifically the sections on `tf.keras.Model` construction, layer introspection, and available pre-trained models. Tutorials provided on the official TensorFlow website are another useful source for additional insights. Additionally, understanding the concept of computational graphs within TensorFlow is crucial for fully grasping the concepts related to embedding extraction and model manipulation. These can be explored in the TensorFlow official guides, particularly those related to graph construction and model customization. Exploring open-source projects on GitHub that utilize TensorFlow can also offer concrete code implementations.
