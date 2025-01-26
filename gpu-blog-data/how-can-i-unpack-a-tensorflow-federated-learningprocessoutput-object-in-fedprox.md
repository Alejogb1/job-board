---
title: "How can I unpack a TensorFlow Federated LearningProcessOutput object in FedProx?"
date: "2025-01-26"
id: "how-can-i-unpack-a-tensorflow-federated-learningprocessoutput-object-in-fedprox"
---

The challenge with extracting information from a TensorFlow Federated (TFF) `tff.learning.templates.LearningProcessOutput` object, particularly within the context of Federated Proximal (FedProx) training, lies in understanding its structure and the asynchronous nature of federated computations. Unlike standard TensorFlow objects, this output is not a simple dictionary or list; it's designed to encapsulate results from a distributed learning process, requiring a specific approach to access individual components. I've frequently encountered this while fine-tuning FedProx models for edge deployment, and I've found a structured method to be crucial.

The `LearningProcessOutput` in TFF, regardless of the specific federated algorithm like FedProx, typically contains three primary attributes: `state`, `metrics`, and `measurements`. However, the structure of `state`, especially, is algorithm-specific. FedProx, being a modification of Federated Averaging, maintains the general structure of having a `model` state but adds a `server_optimizer_state`. These server-side states are critical for tracking algorithm progress, and they are not directly accessible as Python variables.

The `state` element encapsulates the model weights, server optimizer variables (specific to FedProx), and potentially other algorithm-specific data. The `metrics` element holds the aggregation of client-side metrics such as loss and accuracy, computed during training. Finally, the `measurements` element stores per-round statistics, which could include things like the number of clients participating in that specific round or various debug measurements specific to the algorithm. These metrics and measurements are computed and aggregated *across* clients and made available on the server after the round completes.

To access these elements from a returned `LearningProcessOutput`, one must navigate the structure. It is important to realize that the underlying data is frequently structured as `OrderedDict`s. The approach involves accessing the output as a namedtuple. For example, if you have a `LearningProcessOutput` stored in a variable called `output`, you would access its constituent elements like so: `output.state`, `output.metrics`, and `output.measurements`. You must then delve into these attributes to further extract the information you require.

For the `state` component in a FedProx environment, accessing the model weights directly often requires diving deeper. Typically, the model weights are part of the `state.model` attribute. If using `tff.learning.Model` and `tff.learning.optimizers`, the weights are organized under a `trainable` attribute. Consequently, the final sequence of accesses would resemble: `output.state.model.trainable`, where `output` refers to the `LearningProcessOutput`. The output of this, however, is not immediately usable as numpy arrays or simple tensors. You are provided `tff.structure.Struct` which needs to be unwrapped or converted in a TensorFlow session.

The server optimizer state, also part of the `state` in FedProx, is similarly accessed via `output.state.server_optimizer_state`. This state keeps track of parameters specific to the optimizer being used. The type and content of this depend on the optimizer selected (e.g., `tf.keras.optimizers.Adam`).

Here are some practical code examples to clarify this process:

```python
import tensorflow as tf
import tensorflow_federated as tff

# Example TFF model definition
def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, input_shape=(28*28,), activation='relu'),
      tf.keras.layers.Dense(1)
  ])

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=tf.TensorSpec(shape=(None, 28*28), dtype=tf.float32),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanSquaredError()]
  )

# Example data
example_dataset = tff.simulation.datasets.emnist.load_data()
train_data = example_dataset.create_tf_dataset_from_all_clients().take(10)

# Example FedProx algorithm instantiation
optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
fedprox_process = tff.learning.build_federated_averaging_process(
    model_fn,
    optimizer=optimizer,
    client_optimizer_fn=lambda: client_optimizer,
    server_model_update_aggregation_fn=tff.aggregators.Mean(),
    client_weight_fn=lambda x: 1.0,
    use_experimental_simulation_loop=True # Needed for compatability
    )

# Initialize the FedProx process
state = fedprox_process.initialize()

# Perform one round of training
output = fedprox_process.next(state, train_data)

# Accessing state
state = output.state

# The next step is to extract the model from the state.
model_weights = state.model.trainable

# Accessing the metrics
metrics = output.metrics

# Accessing measurements
measurements = output.measurements

print(f"Model Weights: {model_weights}")
print(f"Metrics: {metrics}")
print(f"Measurements: {measurements}")
```
**Explanation of the above Code**
The example sets up a very simple FedProx environment with dummy model and datasets. It then steps through one iteration of FedProx. We can then access and print the attributes of the `LearningProcessOutput`. Note that `model_weights` is still a complex structure and it may require further unwrapping depending on what you intend to do with these model parameters.

```python
# Example 2: Accessing specific metric values
import tensorflow as tf
import tensorflow_federated as tff

# Assume the previous model_fn, optimizer, client_optimizer, fedprox_process, data, and state are defined

# Perform a few rounds of training (assuming loop)
for _ in range(3):
    output = fedprox_process.next(state, train_data)
    state = output.state
    metrics = output.metrics
    print(f"Round Metrics: {metrics}")

    # Extract specific metric
    training_loss = metrics['mean_squared_error']
    print(f"Training Loss: {training_loss}")
```
**Explanation of the above Code**

This example builds upon the first by iterating through a few steps and demonstrates accessing a *specific* metric value, "mean_squared_error" in this case. It is important to know the string key corresponding to the specific metric you wish to access. The `metrics` object is essentially a dictionary-like structure where the keys are the metric names as defined in your `model_fn`.

```python
# Example 3: Extracting and working with model weights
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

# Assume the previous model_fn, optimizer, client_optimizer, fedprox_process, data, and state are defined

# Perform one round of training
output = fedprox_process.next(state, train_data)

# Extract model weights
model_weights = output.state.model.trainable

# Convert to numpy arrays
with tf.compat.v1.Session() as sess:
    np_weights = sess.run([x.numpy() for x in model_weights])


print(f"Numpy Model Weights shape of first weight matrix: {np_weights[0].shape}") # Print shape of the first layer's weights

# Work with np_weights...
```
**Explanation of the above Code**

This final example builds on the previous to show you how to get usable numpy arrays from the model weights. Note that this requires an active Tensorflow session. Model weights are still a complicated structure, but you can now use `numpy` to do computations or visualizations.

In summary, accessing information within a `LearningProcessOutput` from FedProx requires understanding the underlying structure. One must traverse the `state`, `metrics`, and `measurements` attributes to extract relevant data. The `state` element is particularly crucial for FedProx as it holds the server-side optimizer state and model weights. The provided code examples offer practical approaches to accomplishing this. Remember to use a TensorFlow session when you wish to convert `tf.Tensor` to a `numpy` array.
For further learning, review the official TensorFlow Federated documentation on `tff.learning.templates.LearningProcessOutput`, `tff.learning.build_federated_averaging_process`, and the TFF tutorials. These resources provide a comprehensive understanding of the federated learning APIs and should be your first stop. Familiarize yourself with concepts like `tff.learning.Model`, `tff.learning.optimizers`, and `tff.aggregators` as these are essential for correctly setting up and working with TFF. I also strongly encourage reviewing examples that use FedProx or similar methods from the TFF documentation. Lastly, deep understanding of how TensorFlow itself functions will ultimately help you when debugging and working with TFF.
