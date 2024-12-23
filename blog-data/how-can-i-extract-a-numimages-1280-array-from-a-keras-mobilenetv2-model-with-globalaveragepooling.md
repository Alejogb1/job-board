---
title: "How can I extract a (num_images, 1280) array from a Keras MobileNetv2 model with GlobalAveragePooling?"
date: "2024-12-23"
id: "how-can-i-extract-a-numimages-1280-array-from-a-keras-mobilenetv2-model-with-globalaveragepooling"
---

Alright, let's tackle this. I’ve certainly been down this road before, trying to get intermediate outputs from a network, and extracting features specifically after a `GlobalAveragePooling2D` layer can feel a little tricky at first. You're essentially looking to bypass the final classification layers and snag the feature representation just before the model's decision-making stage. It’s a very common scenario when you want to use pre-trained models for transfer learning or feature extraction. So, let’s unpack this systematically.

The key idea here is that we're not interested in the model's output probabilities; we want the activations from the layer *before* the classification head. In the case of MobileNetv2 with `GlobalAveragePooling2D`, this means we want the tensor coming out of the global average pooling operation. This tensor, typically a (batch_size, 1280) shape, encapsulates the learned features of the input images. The 1280 here is specific to MobileNetv2 and represents the number of channels (or filters) after the convolutional layers and the subsequent pooling operation.

Let's explore how you can accomplish this with a few approaches, each with varying degrees of flexibility:

**Method 1: Creating a New Model with a Specified Output Layer**

This method is often the cleanest and most explicit way to grab an intermediate tensor. We effectively redefine our Keras model to only go up to the global average pooling layer. Here’s a snippet:

```python
import tensorflow as tf
from tensorflow import keras

def extract_features_method1(images):
  base_model = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
  intermediate_layer_model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
  features = intermediate_layer_model.predict(images)
  return features

# Example Usage:
if __name__ == "__main__":
    import numpy as np

    num_images = 5
    dummy_images = np.random.rand(num_images, 224, 224, 3) # Example dummy images
    extracted_features = extract_features_method1(dummy_images)
    print(f"Extracted feature shape: {extracted_features.shape}")
    assert extracted_features.shape == (num_images, 1280) #check expected shape
```

In this code:

1.  We load the pre-trained MobileNetv2 model without its classification head (`include_top=False`). This gives us the convolutional base of the network.

2.  We then create a new `keras.Model`. The `inputs` argument remains the same as the original MobileNetv2 model, but the `outputs` are set to the output of the `global_average_pooling2d` layer, which can be found using the `get_layer` method. The naming of layers can be investigated using model.summary().

3.  Finally, we call `predict` on our new `intermediate_layer_model`, giving us the desired (num_images, 1280) array of extracted features.

**Method 2: Using Functional API for Dynamic Feature Extraction**

The functional API provides a more direct way to access layers and their outputs. It's particularly useful if you’re working with more complex or custom architectures, where the naming convention can be more inconsistent. Here’s an implementation:

```python
import tensorflow as tf
from tensorflow import keras

def extract_features_method2(images):
  base_model = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
  gap_output = base_model.get_layer('global_average_pooling2d').output
  intermediate_layer_model = keras.Model(inputs=base_model.input, outputs=gap_output)
  features = intermediate_layer_model.predict(images)
  return features

# Example usage
if __name__ == "__main__":
  import numpy as np

  num_images = 5
  dummy_images = np.random.rand(num_images, 224, 224, 3) # Example dummy images
  extracted_features = extract_features_method2(dummy_images)
  print(f"Extracted feature shape: {extracted_features.shape}")
  assert extracted_features.shape == (num_images, 1280) #check expected shape
```

This is functionally equivalent to the previous method but separates the extraction of the layer output from the new model definition, which can improve readability in complex workflows. We first get the output tensor from the `global_average_pooling2d` layer and then construct our new model using that specific tensor as the output.

**Method 3: Feature extraction during a model fit**

Now, you might think, "Okay, these are good, but what if I need these features as part of the training process, not just inference?". Let’s tackle that scenario. Rather than predicting after the training, we can create an additional output in the training model that returns both the prediction and the features from the intermediate layer:

```python
import tensorflow as tf
from tensorflow import keras

def create_model_with_features():
  base_model = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
  gap_output = base_model.get_layer('global_average_pooling2d').output
  prediction = keras.layers.Dense(10, activation='softmax')(gap_output) # add classification head for this example
  feature_model = keras.Model(inputs=base_model.input, outputs=[prediction, gap_output])
  return feature_model


# Example usage during training
if __name__ == "__main__":
  import numpy as np

  num_images = 50
  dummy_images = np.random.rand(num_images, 224, 224, 3)
  dummy_labels = np.random.randint(0, 10, size=(num_images))
  dummy_one_hot_labels = tf.keras.utils.to_categorical(dummy_labels, num_classes=10)

  model_with_features = create_model_with_features()
  model_with_features.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  history = model_with_features.fit(dummy_images, [dummy_one_hot_labels, np.zeros((num_images,1280))], epochs=5, verbose=0)
  predictions_with_features = model_with_features.predict(dummy_images)

  predictions = predictions_with_features[0]
  features = predictions_with_features[1]

  print(f"Predictions shape: {predictions.shape}")
  print(f"Extracted feature shape: {features.shape}")
  assert features.shape == (num_images, 1280)
```

In this instance:

1.  We construct our base MobileNetv2 model, again without the top layers.
2.  We explicitly create a new output dense layer which will act as the new classification layer for the model.
3.  Then, within the new model, we define two outputs, first the prediction, and second the output of `global_average_pooling2d`.
4. We then proceed to train the model as normal. Note that when fitting, we provide zeros for the second output as there is no loss function defined. We also receive this tensor as part of the models predictions.

**Further Learning**

To deepen your understanding of these concepts, I'd highly recommend diving into the following resources:

*   **"Deep Learning with Python" by François Chollet:** This book is a comprehensive guide to using Keras and covers intermediate layer extraction quite effectively. It provides a solid foundational understanding of model manipulation in keras.

*   **The official Keras documentation:** The keras API is well documented. Specifically, you should look into the `Model` class, layer outputs, and layer retrieval methods. Exploring this will enhance your understanding significantly.

*   **Research papers on Transfer Learning:** If your goal is transfer learning you should have a good grasp of it. Papers like "How transferable are features in deep neural networks" can be useful in understanding the concepts of feature extraction and transfer learning.

In conclusion, extracting that (num\_images, 1280) tensor from MobileNetv2 after the global average pooling isn’t difficult, it’s about understanding the structure of your model and how to properly redefine it (or re-purpose the output) to get the specific information you need. Each of the presented methods has advantages depending on your exact scenario, however, Method 1 is generally the most approachable. This approach should work well for you.
