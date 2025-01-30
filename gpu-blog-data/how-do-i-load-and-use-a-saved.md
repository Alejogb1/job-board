---
title: "How do I load and use a saved TensorFlow model?"
date: "2025-01-30"
id: "how-do-i-load-and-use-a-saved"
---
TensorFlow's model saving mechanism produces a directory containing model architecture, weights, and sometimes even the optimizer state, which can be loaded to reconstruct the model and continue training or conduct inference. Loading and utilizing these saved models requires navigating a structured API, not simply accessing individual files. In my experience, spanning several large-scale machine learning projects involving image classification and natural language processing at my former data science consultancy, I've frequently encountered developers misinterpreting the saved format, resulting in frustrating errors. Therefore, a solid understanding of TensorFlow's `SavedModel` format and the associated loading functions is crucial.

The process hinges on the `tf.saved_model` module, specifically its `load()` function. The `SavedModel` format isn't just a collection of weights; it encompasses the computation graph, allowing TensorFlow to reconstruct the model without the original source code. This is a significant distinction from simpler formats like pickled model files where the underlying code's availability is implicit. TensorFlow saves a model as a directory, including files like `saved_model.pb`, which contains the serialized graph definition, and variables directories containing weights. This directory is the entity you must specify to the `tf.saved_model.load()` function.

The loaded model returns a `tf.train.Checkpoint` object. This object acts as a handle to the reconstructed model and any associated functions that were marked as signature during the save operation. These saved signatures provide a convenient and standardized way to execute the model, often including the forward pass or other custom operations. Without explicitly defined signatures, accessing individual layers and operations becomes complex.

Let me illustrate with three distinct examples demonstrating varied use cases: one showing basic inference, a second demonstrating how to use the model as part of a larger model (transfer learning) and a third demonstrating continuing training from a saved model.

**Example 1: Basic Inference**

In my work on a plant pathology project, we trained a convolutional neural network (CNN) to classify images of diseased leaves. We saved the model after a phase of training. To use this for classifying new unseen images the following code was used:

```python
import tensorflow as tf
import numpy as np

# Directory where the model is saved
model_directory = 'path/to/saved_model'

# Load the saved model. Assumes the save process
# used a signature named 'serving_default'.
loaded_model = tf.saved_model.load(model_directory)

# Obtain the inference function from the loaded model
infer_func = loaded_model.signatures['serving_default']

# Assume the input image is a numpy array with appropriate dimensions.
# For a real-world case, this input will have to be adjusted according to the
# model's expected input format.
input_image = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Perform inference
output = infer_func(tf.constant(input_image))

# Access the results (output will be a dictionary)
# The exact name will depend on the output layer name used during model creation
predictions = output['output_layer_name']
predicted_class = tf.argmax(predictions, axis=1)

print("Predicted class:", predicted_class.numpy())
```

The key step here is accessing the inference function through `loaded_model.signatures['serving_default']`. The name `serving_default` is a common name applied when saving the model and it indicates the model's primary functionality. This example assumes the saved model expects an input in the shape of `(1, 224, 224, 3)` which is typical of a CNN model applied to color images. Also the output name `output_layer_name` will need to be adjusted to match the name of the output layer as defined in the original model architecture. The `tf.argmax()` then returns the class with the highest probability from model's output.

**Example 2: Using the Model for Transfer Learning**

In a separate instance, I was tasked with developing a model to distinguish between different types of medical scans. Rather than training a large model from scratch, we chose to leverage a pre-trained model, a common transfer learning approach. The following code shows how we loaded part of a model to use as a feature extractor:

```python
import tensorflow as tf
from tensorflow.keras import layers

# Directory where the model is saved
model_directory = 'path/to/pretrained_model'

# Load the entire saved model
loaded_model = tf.saved_model.load(model_directory)

# Access a specific layer from the loaded model using its keras layer name.
# This requires the model to have been defined using Keras.
feature_extractor = loaded_model.submodules.my_intermediate_layer #replace my_intermediate_layer with the actual layer name

# Define a new model that uses the feature extractor and adds a custom layer on top
input_tensor = layers.Input(shape=(224, 224, 3))
extracted_features = feature_extractor(input_tensor)
output_tensor = layers.Dense(num_classes, activation='softmax')(extracted_features) # num_classes should be replaced with the actual number
new_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

# Optionally, fine-tune the new model on your task
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare input data and labels as needed
input_data = tf.random.normal((10, 224, 224, 3))
labels = tf.random.uniform((10,), minval=0, maxval=num_classes, dtype=tf.int32)
labels = tf.one_hot(labels, num_classes)

new_model.fit(input_data, labels, epochs=10)
```

Here, the `loaded_model.submodules` member allows access to any Keras layers in the original saved model by their layer names. We isolate a specific layer `my_intermediate_layer` that serves as the starting point for our new model. This new model can then be trained further. It demonstrates how one can extract relevant learned features from a previous model without access to the original source code. It assumes the original model has Keras layers for easy extraction. This is a powerful technique, especially if the underlying model was large or expensive to train.

**Example 3: Resuming Training**

During a large image classification project, we had to interrupt training after several epochs. The next code shows how the checkpointed model was loaded and used to resume the training.

```python
import tensorflow as tf

# Directory where the model is saved
model_directory = 'path/to/saved_model'

# Load the saved model.
loaded_model = tf.saved_model.load(model_directory)

# Load the model's variables.
# This assumes that the 'variables' directory exists in the saved model directory
checkpoint = tf.train.Checkpoint(model=loaded_model)
checkpoint.restore(tf.train.latest_checkpoint(model_directory))

# Get the Keras Model from the loaded object, as needed.
if hasattr(loaded_model, "model"): # check that the saved model was of the Keras type.
    keras_model = loaded_model.model
else:
  raise Exception("The Saved Model does not seem to have been a Keras Model")

# Access the previously saved training state such as the optimizer.
# The specific details here depend on how the original training was implemented.
# It requires that the optimizer be saved during the checkpointing.

# Get the input dataset
input_data = tf.random.normal((10, 224, 224, 3))
labels = tf.random.uniform((10,), minval=0, maxval=num_classes, dtype=tf.int32)
labels = tf.one_hot(labels, num_classes)

# Ensure that the loaded model is compiled.
keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training
keras_model.fit(input_data, labels, epochs=10)
```

This code demonstrates how to load not only the architecture and weights but also the optimizer state if it was saved during the checkpointing. This allows for a seamless continuation of training without having to start from the initial random weights. The crucial part is `checkpoint.restore()`, which restores the previously stored training state, including the model's variables and any optimizer specific settings if saved. This code assumes the model was initially a Keras model and it also assumes it was saved in a way that it could be easily resumed. For example, if the saved model was not originally a Keras model then access to the `keras_model` member will fail.

For further exploration of `SavedModel` functionality, I recommend consulting TensorFlow's official documentation concerning the `tf.saved_model` module, and the `tf.train.Checkpoint` object. The book "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" (Géron) provides extensive practical examples on model saving and loading. For additional theoretical background, reading the original TensorFlow white papers on graph computation may be valuable. Understanding the underlying structure of the `SavedModel` format allows for proper usage of a model beyond its original training context. I hope this technical breakdown provides a solid foundation for you to load and apply pre-trained TensorFlow models.
