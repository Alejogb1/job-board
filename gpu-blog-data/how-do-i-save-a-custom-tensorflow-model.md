---
title: "How do I save a custom TensorFlow model?"
date: "2025-01-30"
id: "how-do-i-save-a-custom-tensorflow-model"
---
When working with TensorFlow, reliably saving custom models—those incorporating layers and logic beyond pre-built architectures—requires understanding the underlying serialization mechanisms. I've spent significant time wrestling with this, particularly when deploying models to resource-constrained environments. The key isn't simply saving the weights; it's preserving the entire model definition, including custom layers, loss functions, and training logic. This ensures consistent behavior when loading the model later, avoiding frustrating discrepancies.

The primary method for saving TensorFlow models is through the `tf.saved_model` API. This API exports models in a standardized format containing the model's computational graph, variables, and metadata. It supports both the `Estimator` API and the newer Keras API, which is now the preferred approach for building and training models in TensorFlow 2.x and beyond.  I'll focus on the Keras approach, as it offers greater flexibility and aligns with modern deep learning practices.

Saving a model in `tf.saved_model` format creates a directory. Inside, you'll find:

*   **`saved_model.pb`**: This protobuf file stores the serialized model graph structure, including layer connections and their parameters.
*   **`variables` subdirectory**:  This folder contains the model's trained weights and biases in binary format. The `variables.data-*` and `variables.index` files store the actual tensors.
*   **`assets` subdirectory**: This directory can hold additional data, such as vocabulary files, if your model requires them.

When saving, you're essentially writing all these components to disk. When loading, TensorFlow reconstructs the model from these files, essentially "rehydrating" the entire entity. This contrasts with approaches that only save parameter weights, necessitating manual reconstruction of the model graph, a process prone to error.

I've often seen newcomers try to save the output of model.get_weights() and later reconstruct models around those weights manually. This approach is brittle and susceptible to versioning issues when code changes. `tf.saved_model` avoids this by preserving the entire architecture alongside the learned parameters.

Now, let's delve into a few code examples, demonstrating different scenarios.

**Example 1: Saving a basic Keras Sequential Model.**

```python
import tensorflow as tf

# Define a simple sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
import numpy as np # Use numpy to make data that doesn't cause runtime issues.
dummy_data = np.random.rand(100,784) # Create random data.
dummy_labels = np.random.randint(0,9,100) # Create random labels.
model.fit(dummy_data, dummy_labels, epochs=2) # Train with dummy data.

# Save the model to a directory named 'my_basic_model'
model.save('my_basic_model')
```

This example saves a basic, fully connected sequential model.  After building and training on dummy data (for demonstration), the `model.save()` function is invoked.  The `'my_basic_model'` argument specifies the directory where the serialized model structure and weights are stored. When I first encountered this I was surprised it was this simple. My previous experiences with other frameworks had much more involved processes to do the same operation. This example highlights the ease of use offered by the Keras API for saving models. Upon running, `my_basic_model` directory is created, containing the `saved_model.pb`, `variables` directory with data, and the `assets` directory (which will be empty here because we did not save any assets).

**Example 2: Saving a Keras Model with Custom Layers.**

```python
import tensorflow as tf

# Define a custom layer
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
      self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
      self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Build a model with custom layer
inputs = tf.keras.Input(shape=(10,))
x = CustomDense(64)(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
dummy_data = np.random.rand(100,10) # Generate dummy data.
dummy_labels = np.random.randint(0,1,100) # Generate dummy labels.
model.fit(dummy_data, dummy_labels, epochs=2) # Train with dummy data.

# Save the model with custom layer
model.save('my_custom_model')
```

This example is a bit more nuanced.  It demonstrates how `tf.saved_model` handles custom layers.  Here, `CustomDense` is a custom linear transformation that inherits from `tf.keras.layers.Layer`. The key point is that `tf.saved_model` preserves the layer *definition* not just its weights. This is crucial because the `call` method of our custom layer defines how the layer will execute. The `build` method defines the weights and biases. Upon saving, the `saved_model.pb` file encodes all of this information, allowing TensorFlow to reconstruct and use the custom layer when the model is loaded later. This is a common issue to grapple with as project complexity grows, and saving custom components in this way proves valuable in the long run. Note that the `fit` method needs data of appropriate dimension and type.

**Example 3: Saving a Model with a Custom Loss Function.**

```python
import tensorflow as tf

# Define a custom loss function
def custom_loss(y_true, y_pred):
  squared_difference = tf.square(y_true - y_pred)
  return tf.reduce_mean(squared_difference)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model with the custom loss
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
dummy_data = np.random.rand(100,1) # Generate dummy data.
dummy_labels = np.random.rand(100,1) # Generate dummy labels.
model.fit(dummy_data, dummy_labels, epochs=2) # Train with dummy data.

# Save the model
model.save('my_model_with_custom_loss')
```

This example shows the preservation of a custom loss function. In some cases it may make sense to create a custom loss function. The `custom_loss` function, defined outside the model architecture, is passed to the `compile` method.  When the model is saved using `model.save()`, TensorFlow internally serializes the custom loss function *definition*.  It does not try to simply capture some output of the loss function; it tracks the actual logic implemented in the function. This means, upon loading, the model will be able to accurately reproduce the training behavior, regardless of whether the code for `custom_loss` exists in the new environment, making deployment and portability easier. In my experience, models using custom losses require an especially careful approach to saving and loading, and this serialization helps address concerns.

**Recommendations**

To further enhance your understanding and proficiency in saving TensorFlow models, I recommend the following resources:

*   **TensorFlow API documentation**: Refer to the official TensorFlow documentation for detailed information on the `tf.saved_model` API and all relevant functions. This documentation often contains the most up-to-date information.
*   **TensorFlow tutorials:** The official TensorFlow tutorials offer guided examples and hands-on exercises to solidify your understanding of saving, loading, and deploying models. Focus particularly on tutorials involving custom components.
*   **Books on deep learning:** Books detailing model building in Tensorflow can provide a more in-depth understanding of the underlying architecture, often containing dedicated sections on model serialization. They can contextualize the practice within broader concepts.

By understanding how `tf.saved_model` captures both the structure and trained parameters of your models, you can improve the reproducibility and deployment of your TensorFlow projects. Always be mindful of the full model definition—not just the weights—when thinking about saving your work.
