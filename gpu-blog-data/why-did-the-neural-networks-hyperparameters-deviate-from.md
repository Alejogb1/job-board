---
title: "Why did the neural network's hyperparameters deviate from expectations when fitting a reduced Boolean function?"
date: "2025-01-30"
id: "why-did-the-neural-networks-hyperparameters-deviate-from"
---
Boolean functions, particularly those with limited input dimensionality, present a deceptively simple training ground for neural networks. My experience in developing a custom hardware accelerator for embedded AI exposed me to scenarios where even seemingly trivial functions, such as a reduced AND operation between two inputs, could lead to unexpected hyperparameter sensitivity during training. The core issue lies not in the network’s inability to *learn* the function but rather in the specific path it takes to do so, and how this path is affected by the learning algorithm's inherent biases and the initialization of the network itself.

Typically, when one considers a boolean function, especially those implementable by a single neuron, a straightforward optimization might be expected, with a relatively direct path to a well-defined global minimum. However, even a small neural network with multiple layers and neurons introduces a non-convex optimization landscape. This is crucial because the optimization algorithms we use, like stochastic gradient descent (SGD) and its variants, are designed for finding *local* minima, not necessarily the global minimum. When the landscape has multiple valleys, the specific hyperparameter values can determine which valley the training process falls into, affecting the convergence trajectory and the network's parameter values. A seemingly "correct" convergence path based on one hyperparameter configuration may not be replicated with a slightly different setup.

Consider the following factors leading to hyperparameter deviations with reduced Boolean functions:

1. **Weight Initialization:** The initial weights assigned to the network have a considerable impact on training. For instance, when using a uniform or Gaussian distribution, these random initial values might push the network into a region of the weight space that is far from the ideal solution for a very simple function. Furthermore, when one uses activation functions, such as ReLU, a specific weight configuration might cause neurons to remain inactive or saturated during the training process. This is particularly true with a small number of samples. If the initial weight configuration pushes the neurons towards saturation on each sample, they may not update correctly. In the case of a two-input AND, the ideal solution involves very specific weights and bias which initially have a low probability of occurring during random initialization.

2. **Learning Rate:** The learning rate controls the step size during optimization. A large learning rate can cause the model to overshoot the optimal weights, oscillating wildly around the ideal solution and potentially preventing convergence. A small learning rate, on the other hand, might result in very slow convergence, possibly settling in a sub-optimal local minimum. This can be problematic when fitting a basic boolean function, as there's an expectation of rapid convergence given the simplicity of the function. A reduced learning rate can also become stuck in plateau regions, where updates are too slow to escape.

3. **Optimizer Selection:** Different optimizers (SGD, Adam, RMSprop) have distinct adaptation properties and sensitivities to the learning rate and other hyperparameters. Adam, for example, has adaptive learning rates for individual parameters, which means that while it often accelerates training it can also make the optimization path sensitive to how the gradients initially look. A basic boolean function may not benefit from this complexity and might even cause instabilities when using more complex optimizers. A basic optimizer like SGD, using a carefully chosen learning rate, can sometimes be more effective when dealing with simple boolean functions.

4. **Batch Size:** This affects the stochasticity of the updates. A small batch size introduces noisy gradient estimations and, as a result, leads to a less stable path during optimization. Conversely, with a Boolean function, the noise introduced by a small batch size can make it more difficult for the network to find the narrow optimal region. When sample size is also small, like a typical boolean function's complete truth table, batch size can drastically alter performance.

5. **Network Architecture:** While it might appear counter-intuitive, the depth and width of the network also affect its behavior. More complex architectures introduce additional parameters and consequently a more complex optimization surface. This means that what could be easily learned using a shallow network might exhibit instability with a deep one even though its approximation ability is superior. With small datasets and trivial functions like Boolean operations, these can become a significant problem.

Here are three code examples using Python and TensorFlow, illustrating these sensitivities:

**Example 1: Varying Learning Rate (SGD)**

```python
import tensorflow as tf
import numpy as np

# Define the boolean AND function truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

def train_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,))
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=1000, verbose=0) # Run silently
    loss, accuracy = model.evaluate(X, y, verbose=0)
    return accuracy

# Test different learning rates
learning_rates = [0.01, 0.1, 1.0]
for lr in learning_rates:
    accuracy = train_model(lr)
    print(f"Learning Rate: {lr}, Accuracy: {accuracy}")
```

In this example, a very basic neural network is trained with different learning rates. One might expect perfect accuracy, since a single neuron with sigmoid activation can represent this Boolean function effectively with proper weights and biases. However, with a large learning rate of 1.0, the optimizer will often fail to converge to 100% accuracy due to oscillations. Conversely, a lower rate of 0.01 might have a slow convergence, requiring additional training epochs. An intermediate value, around 0.1, typically yields near perfect or perfect accuracy much more quickly. This illustrates how learning rate significantly impacts convergence for even this very simple case.

**Example 2: Weight Initialization Variation**

```python
import tensorflow as tf
import numpy as np

# Define the boolean AND function truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

def train_model(initializer):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,), kernel_initializer=initializer)
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=1000, verbose=0)
    loss, accuracy = model.evaluate(X, y, verbose=0)
    return accuracy

# Test different initializers
initializers = [tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
                 tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
                 tf.keras.initializers.Zeros()]

for init in initializers:
    accuracy = train_model(init)
    print(f"Initializer: {init}, Accuracy: {accuracy}")
```

This second example demonstrates how different initializers affect training, even when the other parameters are fixed. With an initializer generating zero weights, the gradients would initially be zero which will prevent effective training. A random initialization allows for a much quicker convergence, but even within the random initialization, different distributions (Uniform vs Normal) will result in varying convergence speeds and even convergence trajectories. These minor alterations lead to non-trivial performance variations, thus illustrating that the very first weights impact final training success.

**Example 3: Batch Size Effect**

```python
import tensorflow as tf
import numpy as np

# Define the boolean AND function truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)


def train_model(batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,))
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=1000, batch_size=batch_size, verbose=0)
    loss, accuracy = model.evaluate(X, y, verbose=0)
    return accuracy

# Test different batch sizes
batch_sizes = [1, 2, 4]

for bs in batch_sizes:
    accuracy = train_model(bs)
    print(f"Batch Size: {bs}, Accuracy: {accuracy}")
```

This final example shows how batch size influences the training, illustrating how different batch sizes affects training stability. When the batch size is 1, the gradient is calculated using a single sample, creating a very noisy update process. This leads to instability and slower convergence. Conversely, batch size 4 calculates the gradients using all training samples at once, resulting in a more accurate estimation and potentially faster training, which might seem counterintuitive due to the small dataset of only 4 samples. This highlights the sensitivity of the gradient updates to batch size.

In conclusion, despite the simplicity of a reduced Boolean function, the optimization process of a neural network is sensitive to a range of hyperparameters, as shown in the preceding code examples and discussions. These deviations underscore that neural network training is a complex stochastic optimization process. It’s not a deterministic process, even when you are solving an apparently trivial problem. This highlights the need to understand the underlying mechanisms and carefully select these hyperparameters based on data, network architecture, and optimizer rather than assuming a universal or "correct" set of values.

For further study, I would suggest focusing on academic publications about the following topics: deep learning optimization theory, non-convex optimization, stochastic gradient descent variants, initialization methods in deep learning, and practical deep learning techniques for tuning hyperparameters. Exploring implementations of optimizers in major libraries, and reading detailed documentation about initializers and activation functions would also be very beneficial. This thorough approach can help explain the sensitivity to the hyperparameters when using neural networks for simple functions.
