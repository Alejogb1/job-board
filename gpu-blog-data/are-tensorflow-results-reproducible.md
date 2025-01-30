---
title: "Are TensorFlow results reproducible?"
date: "2025-01-30"
id: "are-tensorflow-results-reproducible"
---
TensorFlow, while striving for determinism, presents challenges to complete result reproducibility. The stochastic nature of deep learning algorithms, combined with hardware-specific optimizations and software dependencies, makes bit-for-bit identical results across different runs, and certainly different platforms, difficult to guarantee. My experience training models across various GPU clusters and local machines has consistently shown subtle variations, even with seemingly identical configurations. While we can significantly improve reproducibility, achieving absolute parity requires careful attention to multiple contributing factors.

The core issue stems from the inherent randomness introduced during model initialization, data shuffling, and, importantly, the parallel processing of operations. The backpropagation algorithm relies on gradient descent, which uses initial weights as starting points for the optimization process. These weights are often randomly initialized, typically from a uniform or normal distribution. Even if the seed for the random number generator is fixed, differences in hardware architectures or CUDA versions can lead to minute variations in floating-point operations during calculations. These minute variations, accumulated across millions of parameters and numerous training iterations, can eventually lead to perceptible differences in the final model. Similarly, data shuffling, if not carefully managed using a consistent random seed across runs, will expose the model to training data in varying orders, affecting the learning trajectory.

Further complicating matters, TensorFlow leverages parallel processing on GPUs for computational speedup. The order in which concurrent operations are scheduled and executed is often not deterministic. Even when using libraries like NumPy with fixed random seeds, the results of certain functions may not be fully reproducible on different systems. The multi-threading behavior, coupled with CUDA drivers, can introduce non-deterministic behavior regarding reduction operations, further causing slight numerical differences. In essence, while TensorFlow aims for reproducibility, true bitwise determinism is a complex problem, dependent on both algorithmic and platform-specific factors.

Let's delve into some code examples illustrating this, along with strategies to mitigate these challenges.

**Example 1: Setting Seeds and Data Shuffling**

This example showcases how critical setting random seeds is for consistent initializations and dataset processing.

```python
import tensorflow as tf
import numpy as np

# Define a function to train a simple model
def train_model(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Generate dummy data
    num_samples = 100
    features = 10
    X = np.random.rand(num_samples, features).astype(np.float32)
    y = np.random.randint(0, 2, num_samples).astype(np.int32)

    # Create and compile a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(features,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Shuffle data using a consistent seed during training
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(num_samples, seed=seed)
    dataset = dataset.batch(32)

    # Train the model
    model.fit(dataset, epochs=2, verbose=0) # Reduced epochs for brevity

    return model.get_weights()[0][0][0] # Extracting a specific weight for comparison

# Run the training with the same seed
seed_val = 42
weight1 = train_model(seed_val)
weight2 = train_model(seed_val)

# Run the training with a different seed
diff_seed_weight = train_model(123)


print(f"Weight after training with seed {seed_val}: {weight1:.8f}")
print(f"Weight after training with seed {seed_val} again: {weight2:.8f}")
print(f"Weight after training with a different seed: {diff_seed_weight:.8f}")
print(f"Are the weights from the same seed identical?: {np.allclose(weight1, weight2)}")

```

Here, setting `tf.random.set_seed` and `np.random.seed` allows the model to initialize identically across runs if the seed is the same. The `shuffle` method of the TensorFlow dataset also receives the same seed. The output shows that weight extraction for same seed runs are equal. This shows the importance of consistently setting seeds for random number generation when attempting to produce reproducible results, although it is a first step only.

**Example 2: Device-Specific Behavior**

This code demonstrates the differences in behaviour that can appear on different hardware. For this example, I assume two differing CPU types to explain the concept; however, in reality the differences would also be between CPUs, GPUs, TPUs and even different generations of the same hardware, or differing driver versions. For illustration purposes we do not need to run this code directly as the variations do not stem from randomness, but instead from intrinsic hardware characteristics.

```python
import tensorflow as tf
import numpy as np

def matrix_multiply_test(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    a = tf.random.normal((1000, 1000), dtype=tf.float32)
    b = tf.random.normal((1000, 1000), dtype=tf.float32)

    result = tf.matmul(a, b)
    return np.mean(result.numpy())

seed_val = 42
result_cpu_a = matrix_multiply_test(seed_val) # run on CPU-A (hypothetical)
result_cpu_b = matrix_multiply_test(seed_val) # run on CPU-B (hypothetical)

print(f"Mean of matrix multiplication (CPU-A): {result_cpu_a:.8f}")
print(f"Mean of matrix multiplication (CPU-B): {result_cpu_b:.8f}")
print(f"Are the means equal? : {np.allclose(result_cpu_a,result_cpu_b)}")

```

In this fabricated scenario, while the same code is executed with the same seed, the underlying operations on differing CPU architectures may utilize slightly different algorithms for floating-point arithmetic. Even when using IEEE-754 standards, minor deviations in how the calculation is performed by a given processor may lead to slightly different numerical results. This illustrates that even if we take care to seed all the random elements of an operation, the hardware itself might introduce non-determinism.

**Example 3: Configuration for Deterministic Operations**

Here we look at additional configuration changes that can influence determinism.

```python
import tensorflow as tf
import os

# Enable deterministic operations (experimental - be aware of potential performance impacts)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.experimental.enable_op_determinism()


def train_model_with_determinisim(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Generate dummy data
    num_samples = 100
    features = 10
    X = np.random.rand(num_samples, features).astype(np.float32)
    y = np.random.randint(0, 2, num_samples).astype(np.int32)

     # Create and compile a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(features,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Shuffle data using a consistent seed during training
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(num_samples, seed=seed)
    dataset = dataset.batch(32)

    # Train the model
    model.fit(dataset, epochs=2, verbose=0)

    return model.get_weights()[0][0][0] # Extracting a specific weight for comparison


seed_val = 42
weight_det_1 = train_model_with_determinisim(seed_val)
weight_det_2 = train_model_with_determinisim(seed_val)

print(f"Weight after training with determinism enabled: {weight_det_1:.8f}")
print(f"Weight after training with determinism enabled again: {weight_det_2:.8f}")
print(f"Are the weights identical with determinism on?: {np.allclose(weight_det_1, weight_det_2)}")
```

This example illustrates the use of environment variables and a configuration option in TensorFlow to enable deterministic operations. This setting forces certain operations to be executed in a way that should provide reproducible results, but can impact performance. In some cases, a performance penalty may be incurred and the degree to which these settings affect results can vary across different TensorFlow versions and hardware. However, under ideal conditions, these settings can help provide more consistent results, and we can see that the weights in the example will be consistent across multiple runs.

It is important to note that `TF_DETERMINISTIC_OPS` does not guarantee perfect reproducibility. Some operations inherently lack deterministic implementations, and the level of determinism can vary across platforms. This is a pragmatic step to improve results for models that rely on multiple operations.

**Recommendations for further study:**

To better understand this topic, I would recommend researching the following concepts through textbooks, official documentation, or online resources:

*   **Floating-point arithmetic:** Understanding the limitations of floating-point representation in computers, particularly how rounding errors can accumulate and introduce non-determinism.
*   **Parallel computing:** Studying the fundamentals of parallel execution models, including threading and how scheduling and execution can introduce variance.
*   **Random number generation:** Examining the properties of pseudorandom number generators, and the critical importance of proper seeding in reproducible experiments.
*   **IEEE 754 standard:** Understanding the details of the standard for floating-point arithmetic, and the areas where the standard itself may permit slight variation
*   **TensorFlow documentation:** Focusing on the specific sections that detail how to control randomness, hardware specific configurations and deterministic operation settings.

In conclusion, while absolute bit-for-bit reproducibility of TensorFlow results is a nuanced problem, carefully managing random seeds, data preprocessing, and taking advantage of deterministic operation settings can dramatically reduce variations across different runs, facilitating research and model deployment. However, one must be mindful of potential hardware and platform related variations and understand the underlying algorithmic and computational factors that contribute to differences in results.
