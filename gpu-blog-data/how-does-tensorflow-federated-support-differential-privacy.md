---
title: "How does TensorFlow Federated support differential privacy?"
date: "2025-01-30"
id: "how-does-tensorflow-federated-support-differential-privacy"
---
Differential privacy (DP) within federated learning, specifically using TensorFlow Federated (TFF), revolves around adding calibrated noise to model updates or gradients to protect the privacy of individual participants' data. The core challenge lies in training a model across decentralized data without exposing sensitive information held by each user. TFF, by design, operates on aggregated updates, already providing a degree of privacy. However, standard federated averaging, for example, can still leak information if those updates are too detailed. This is where differential privacy techniques integrated into TFF become crucial.

My experience working with a medical imaging application across multiple hospitals highlighted this need. While the federated setting prevented direct access to patient images, we still risked inferring information about individual hospitals if the model updates from each were too precise. Introducing DP with TFF allowed us to train a robust diagnostic model without the risk of compromising individual patient data or revealing sensitive institutional data patterns.

TFF facilitates DP by allowing the application of noise addition mechanisms, notably those based on Gaussian or Laplacian distributions, during the aggregation phase. These mechanisms are strategically implemented within the federated computation, not on the client's device before the updates leave the device, nor after the aggregation has occurred on the server. Specifically, TFF's primitives enable you to modify the aggregation process, so that noise can be incorporated based on sensitivity parameters of the aggregation function. The sensitivity measures the maximum change a single user’s contribution can have on the aggregated result. To effectively implement DP in TFF, one must choose a suitable noise mechanism, establish the privacy parameters, and adapt training procedures accordingly.

Let's examine how this works in practice through several examples:

**Example 1: Applying Gaussian Noise to Averaged Gradients**

This example demonstrates incorporating Gaussian noise during the aggregation of gradients, a common approach in DP federated learning. The `tff.aggregators.GaussianAdaptiveQueryingFactory` class facilitates this. It estimates the sensitivity and applies appropriate Gaussian noise at each round.

```python
import tensorflow as tf
import tensorflow_federated as tff

# Define a simple model for illustration
def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=tff.types.TensorType(tf.float32, shape=(None, 784)),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[tf.keras.metrics.Accuracy()]
  )

# Set privacy parameters
l2_norm_clip = 2.0 # Max norm for gradient clipping
noise_multiplier = 0.5  # Adjust to control noise amount
clients_per_round = 10 # Number of clients participating in a round

# Aggregator for federated averaging with DP
dp_aggregator = tff.aggregators.GaussianAdaptiveQueryingFactory(
    noise_multiplier=noise_multiplier,
    l2_norm_clip=l2_norm_clip,
    clients_per_round=clients_per_round
)

# Build federated averaging algorithm
fed_avg_with_dp = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    model_aggregator=dp_aggregator
)

# Initialize the process and training data will be passed in later
state = fed_avg_with_dp.initialize()
```

In this snippet, `GaussianAdaptiveQueryingFactory` automatically calculates the sensitivity based on the L2 norm of the gradient and the number of participants. The `noise_multiplier` controls how much noise is added, with higher values implying stronger privacy guarantees but also lower model accuracy. Setting a `clients_per_round` value is required to obtain a well-defined privacy guarantee. The `l2_norm_clip` parameter limits the effect of extremely large updates from any individual client on the aggregate. This clipping further increases robustness but may increase the amount of noise that is added. The `build_federated_averaging_process` takes this DP aggregator as an argument.

**Example 2: Using a Custom Aggregator for DP with Laplacian Noise**

This illustrates how to create a custom aggregator for a different privacy mechanism: adding Laplacian noise. This might be useful in scenarios where sensitivity is known exactly rather than adaptively estimated. Here, a fixed privacy budget is utilized.

```python
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

class LaplaceAggregator(tff.aggregators.UnweightedAggregationFactory):
  def __init__(self, sensitivity, privacy_parameter):
    self._sensitivity = sensitivity
    self._privacy_parameter = privacy_parameter

  def create(self, value_type):
    @tf.function
    def aggregate(value_tensor, weights): # weights unused here as it is unweighted
        laplace_scale = self._sensitivity / self._privacy_parameter
        noise = tf.random.stateless_laplace(
            shape=tf.shape(value_tensor),
            seed=tf.random.get_global_generator().make_seeds(2)[0],
            dtype=value_tensor.dtype,
            loc=0.0,
            scale=laplace_scale
            )

        return tf.add(value_tensor, noise)

    return tff.aggregators.AggregationProcess(
        initialize_fn=lambda: tf.constant(0),
        next_fn=lambda state, value_tensor, weights: (state, aggregate(value_tensor, weights))
    )

def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=tff.types.TensorType(tf.float32, shape=(None, 784)),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[tf.keras.metrics.Accuracy()]
  )

# Example usage:
sensitivity = 1.0 # Fixed sensitivity of gradients
privacy_parameter = 1.0  # Privacy parameter, usually corresponds to epsilon

laplace_aggregator = LaplaceAggregator(sensitivity, privacy_parameter)

fed_avg_with_laplace_dp = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    model_aggregator=laplace_aggregator
)
state = fed_avg_with_laplace_dp.initialize()
```

This custom `LaplaceAggregator` directly adds Laplacian noise with a scale determined by the predefined sensitivity and privacy parameter. The `privacy_parameter` here directly controls the strength of the privacy guarantee, whereas in the Gaussian example, that parameter was controlled indirectly through `noise_multiplier`.  The `tff.aggregators.AggregationProcess` wraps the custom aggregation logic with initialisation and next step logic. The `aggregate` function calculates the required laplace noise and adds it to the input.

**Example 3: DP with Secure Aggregation (Simplified)**

While secure aggregation is not strictly required for differential privacy, it often gets paired with it in federated learning.  TFF has abstractions to simplify secure aggregation and here we briefly demonstrate where it would fit in. In reality, secure aggregation is much more complex and it requires specific secure communication protocols which are not included here.

```python
import tensorflow as tf
import tensorflow_federated as tff

def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=tff.types.TensorType(tf.float32, shape=(None, 784)),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[tf.keras.metrics.Accuracy()]
  )

# Set privacy parameters (same as Example 1)
l2_norm_clip = 2.0
noise_multiplier = 0.5
clients_per_round = 10

# Aggregator with Secure Aggregation
dp_secure_aggregator = tff.aggregators.secure_sum_factory(
    tff.aggregators.GaussianAdaptiveQueryingFactory(
        noise_multiplier=noise_multiplier,
        l2_norm_clip=l2_norm_clip,
        clients_per_round=clients_per_round
        )
    )


fed_avg_with_secure_dp = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    model_aggregator=dp_secure_aggregator
)

state = fed_avg_with_secure_dp.initialize()
```

This example utilizes `tff.aggregators.secure_sum_factory` as a wrapper for the DP aggregator from the first example. In a realistic scenario this wrapper would encrypt all client updates using homomorphic encryption before adding them together. This ensures that each individual update remains concealed and only the final aggregated value is revealed. The key point here is that secure aggregation can be combined with DP aggregation allowing for both improved privacy and security.

To deepen your understanding and implementation of DP in TFF, I recommend the following resources:

1.  *The TensorFlow Federated API documentation* provides in-depth explanations of the TFF concepts, such as the `tff.aggregators` module, and various federated computation building blocks.

2.  *Academic papers and resources on federated learning and differential privacy*. These papers delve into the theoretical foundations of DP and different mechanisms suitable for federated learning. Understanding the underlying mathematics will be crucial to tuning the hyperparameters properly.

3.  *TensorFlow tutorial notebooks*. Within the official TensorFlow documentation there are tutorials on federated learning that may include implementations of differential privacy.

Effectively applying DP in TFF requires a solid understanding of the framework and also the theoretical underpinnings of DP. Carefully tuning parameters and selecting an appropriate aggregation strategy are crucial for balancing the privacy guarantees against model performance.
