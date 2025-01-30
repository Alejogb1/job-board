---
title: "How do I print KerasTensor input values?"
date: "2025-01-30"
id: "how-do-i-print-kerastensor-input-values"
---
The direct access to the numerical values held within a `KerasTensor` during model execution is not straightforward because they represent symbolic placeholders within the TensorFlow computational graph, not concrete values. These tensors are symbolic descriptions of operations, not containers of data, until the graph is executed either during training or inference. I've spent considerable time debugging complex custom layers and loss functions, and the approach to inspecting these values often requires a shift in how we think about tensor manipulation within a Keras workflow. The immediate goal isn't 'printing' the values *directly*, but instead, crafting a mechanism that allows us to inspect their computed results at specific points in the model's processing pipeline.

The challenge stems from TensorFlow's delayed execution paradigm. When you define a Keras model, you're primarily constructing a directed acyclic graph describing computations. This graph remains abstract until fed with data. Therefore, attempting to print a `KerasTensor` object will only yield its symbolic representation (e.g., `<KerasTensor: shape=(None, 32) dtype=float32, name=dense_1/BiasAdd:0>`), not its numerical contents. To obtain the numerical data, I rely upon techniques that effectively 'eavesdrop' on the actual computation, either by creating a temporary output point or by using debugging facilities available in TensorFlow.

Here are three common techniques I employ, each with specific use cases:

**Technique 1: Creating an Intermediate Model**

This technique involves extracting a section of the original model and wrapping it within a temporary Keras `Model` object. This allows us to feed data into the intermediate model and directly inspect its output, which will be the `KerasTensor` values we are interested in. The primary advantage is targeted inspection without altering the main training/inference loop. This is my go-to method when I need to understand tensor shapes, ranges, or distributions in complex multi-stage layers.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Construct a simple model (for demonstration)
input_shape = (10,)
inputs = keras.Input(shape=input_shape)
x = layers.Dense(32, activation='relu')(inputs)
output = layers.Dense(10, activation='softmax')(x)
full_model = keras.Model(inputs=inputs, outputs=output)

# Target the intermediate tensor (x after the first dense layer)
intermediate_tensor = x

# Create a temporary model that outputs the intermediate tensor
intermediate_model = keras.Model(inputs=full_model.input, outputs=intermediate_tensor)

# Generate dummy data
dummy_input = np.random.random((2, 10))

# Predict with the intermediate model
intermediate_output = intermediate_model.predict(dummy_input)

# Print the intermediate output values
print("Intermediate Tensor values:\n", intermediate_output)
print("Intermediate tensor shape:", intermediate_output.shape)
```

In this example, the `intermediate_model` is created to isolate the output of the first `Dense` layer (relu activation). `dummy_input` represents the input of the full model. By using `intermediate_model.predict`, I explicitly execute the sub-graph leading up to the intermediate tensor and access the numerical result, which is then printed to the console. The `print` statements now reveal the actual values of the KerasTensor, along with their shape, after processing through the initial `Dense` layer. This provides concrete, runtime numeric information.

**Technique 2: Custom Callback for Debugging During Training**

When inspecting tensors during training, a custom Keras `Callback` class proves useful. This mechanism hooks into the training loop at various points, allowing the extraction and inspection of specific tensors. I often employ this approach when I am debugging loss functions or evaluating the impact of custom gradients, requiring step-by-step visibility of specific intermediate calculations. The `on_batch_begin` or `on_batch_end` methods can be especially helpful in such cases.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class TensorPrintingCallback(keras.callbacks.Callback):
    def __init__(self, tensor_to_print):
        super(TensorPrintingCallback, self).__init__()
        self.tensor_to_print = tensor_to_print

    def on_batch_begin(self, batch, logs=None):
        # Obtain the value of the tensor by feeding dummy data through a submodel
        intermediate_model = keras.Model(inputs=self.model.input, outputs=self.tensor_to_print)
        dummy_input = np.random.random((1, 10)) # Assuming same input shape (adjust as needed)
        tensor_value = intermediate_model.predict(dummy_input, verbose=0) # Supress verbosity of predict
        print(f"\nBatch: {batch}, Tensor Value:\n {tensor_value}")


# Define a simple model for training
input_shape = (10,)
inputs = keras.Input(shape=input_shape)
x = layers.Dense(32, activation='relu')(inputs)
output = layers.Dense(1, activation='sigmoid')(x) # Sigmoid instead of Softmax for simplified binary classification
model = keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Data (replace with actual data)
dummy_data_input = np.random.random((100, 10))
dummy_data_output = np.random.randint(0, 2, (100, 1))  #Binary classification output

# Instantiate the callback
callback = TensorPrintingCallback(tensor_to_print=x) #Print the same tensor as before

# Train the model using dummy data with the tensor printing callback
model.fit(dummy_data_input, dummy_data_output, epochs=2, batch_size=32, callbacks=[callback])

```
This example defines `TensorPrintingCallback`, which, during `on_batch_begin`, uses a similar submodel technique as in the first method to extract the value of tensor `x` before any gradient updates within a batch. The dummy data ensures that the submodel receives a valid input. The verbose=0 flag is to prevent the prediction from printing batch progression. This permits a real-time, training-context view of the inspected tensor.

**Technique 3: Using TensorFlow's Eager Execution and `tf.print`**

TensorFlow 2's eager execution mode simplifies certain debugging procedures. With `tf.function`, we can make computations run like standard Python functions, allowing easier examination of the values. I find this most beneficial when debugging the internals of a custom layer function that operates directly on tensors, before they are part of a larger graph. TensorFlow's `tf.print` offers direct output capabilities in this context, unlike standard Python print.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Activate Eager execution
tf.config.run_functions_eagerly(True)

@tf.function
def process_tensor(inputs):
  x = layers.Dense(32, activation='relu')(inputs)
  tf.print("Tensor values:", x)
  return layers.Dense(10, activation='softmax')(x)

# Create a simple model that wraps around the eager function
input_shape = (10,)
inputs = keras.Input(shape=input_shape)
output = process_tensor(inputs)
model = keras.Model(inputs=inputs, outputs=output)


# Dummy Data
dummy_input = np.random.random((2, 10))

# Perform a single prediction
output_tensor = model.predict(dummy_input)


# Reset to default value.  Be careful with the tf.config API and when calling it
tf.config.run_functions_eagerly(False)

```
Here, the `process_tensor` function is decorated with `tf.function`. This means it will run in an eager execution setting. The `tf.print` within will reveal the numerical data of the `x` tensor during each execution of the `process_tensor`. The prediction operation `model.predict` triggers the eager execution. The `tf.config.run_functions_eagerly(False)` line restores default non-eager mode. I strongly suggest avoiding this approach during high performance model training, as the eagerness impacts overall efficiency.

In summary, inspecting `KerasTensor` numerical data necessitates a technique where you explicitly extract the computation result. Building intermediate Keras Models, implementing custom training callbacks, and leveraging TensorFlow's eager execution mode offer viable methods for this. I advise a layered approach: start with simple intermediate models for quick checks, then graduate to callbacks for training-context inspection, and lastly, consider eager execution when scrutinizing custom tensor operations.

For further in-depth understanding, I would recommend exploring the TensorFlow documentation on `tf.function`, Keras Models, and custom Callbacks. Reading papers describing TensorFlow's graph compilation and execution models can also prove beneficial for understanding the subtle nuances of Keras Tensor management. The official TensorFlow tutorials often provide use-case examples of these concepts.
