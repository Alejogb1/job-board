---
title: "How do I print all activation shapes in a TensorFlow 2 Keras model?"
date: "2025-01-30"
id: "how-do-i-print-all-activation-shapes-in"
---
Determining the activation shapes within a TensorFlow 2 Keras model requires a nuanced understanding of the model's internal structure and the data flow through its layers.  My experience optimizing large-scale convolutional neural networks for image recognition frequently necessitates this level of introspection, particularly when debugging performance bottlenecks or architectural issues.  The key lies in leveraging the model's inherent capabilities for layer introspection and leveraging the `tf.TensorShape` object.  Directly accessing activation shapes during the model's forward pass is generally not directly possible, requiring an indirect approach.  Instead, we must utilize the model's built-in methods and potentially custom layer hooks to capture these shapes.

**1. Explanation:**

The lack of a single, built-in function to retrieve all activation shapes stems from the dynamic nature of TensorFlow's computation graph.  While the model's architecture defines the *potential* shapes, the actual shapes realized during execution depend on the input data.  Therefore, obtaining these shapes requires either intercepting the activations within the layers or traversing the model's layers and inferring shapes based on layer configurations and input tensor shapes.  The most reliable method involves creating a custom callback or using a layer-wise inspection during model building.  This approach provides direct access to the output tensors of each layer, allowing precise shape retrieval.

**2. Code Examples:**

**Example 1: Using a Custom Callback**

This approach utilizes a custom Keras callback to log activation shapes after each batch. This is particularly useful for models with large inputs where inspecting the entire model at once is impractical or computationally expensive.

```python
import tensorflow as tf
from tensorflow import keras

class ActivationShapeLogger(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            try:
                output_shape = layer.output_shape
                print(f"Layer '{layer.name}': Output Shape - {output_shape}")
            except AttributeError:
                # Handle layers without output_shape attribute
                print(f"Layer '{layer.name}': Output shape not directly accessible.")

# Example model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Training with callback
model.fit(
    x=tf.random.normal((100,10)),
    y=tf.random.normal((100,10)),
    epochs=1,
    callbacks=[ActivationShapeLogger()]
)
```

This code defines a callback that iterates through the model's layers after each training batch.  It attempts to access `layer.output_shape` for each layer.  The `try-except` block gracefully handles layers which might not directly expose this attribute (e.g., custom layers).  The output shows the shape of the activation after each layer for every batch.


**Example 2:  Layer-wise Inspection during Model Building**

This method involves iterating through the layers during model construction.  It leverages the `input_shape` of each layer and the layer's configuration to infer the output shape. This method offers a static view of the shapes *before* training commences, useful for verifying model architecture before committing to expensive training runs.

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


def get_layer_output_shapes(model):
    shapes = []
    input_shape = model.input_shape
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            output_shape = (input_shape[0], layer.units)
        # Add similar logic for convolutional or other layer types here.
        else:
            output_shape = "Shape inference not implemented for this layer type"
        shapes.append((layer.name, output_shape))
        input_shape = output_shape
    return shapes

output_shapes = get_layer_output_shapes(model)
for layer_name, shape in output_shapes:
    print(f"Layer '{layer_name}': Inferred Output Shape - {shape}")
```

This example demonstrates inferring output shapes based on layer type and input shape.  The function `get_layer_output_shapes` iterates through the layers.  Crucially, it currently only supports `Dense` layers;  more comprehensive shape inference would require handling other layer types (convolutional, recurrent, etc.) individually, a process dependent upon the layers' specific parameterization.


**Example 3: Using a Functional API Model and Lambda Layers for explicit shape logging:**

This approach leverages the flexibility of the Keras Functional API to explicitly log shapes using Lambda layers.  It provides precise control and clarity over the process, although it increases the model's complexity.

```python
import tensorflow as tf
from tensorflow import keras

def log_shape(x):
    print(f"Layer Output Shape: {x.shape}")
    return x

inputs = keras.Input(shape=(10,))
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Lambda(log_shape)(x) #Explicit shape logging
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Lambda(log_shape)(x) #Explicit shape logging
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

#Dummy data for shape inference
model.predict(tf.random.normal((1,10)))
```

This uses `Lambda` layers with the custom `log_shape` function to explicitly print the shape of the tensor passed through.  This requires adding a `Lambda` layer after each layer whose activation shape needs to be printed.  The `model.predict` call triggers the forward pass and the shape logging within the lambda layers.


**3. Resource Recommendations:**

The TensorFlow documentation, specifically the sections detailing the Keras API and custom callbacks, provides invaluable information.  Furthermore, textbooks on deep learning architectures and practical implementations offer comprehensive background on model construction and data flow.  Reviewing code examples from established deep learning repositories can provide further insight into common implementation patterns.  Finally, exploring the TensorFlow source code itself can prove extremely valuable for understanding the underlying mechanics of shape inference and tensor manipulation.
