---
title: "How does batch normalization work in Keras?"
date: "2025-01-30"
id: "how-does-batch-normalization-work-in-keras"
---
Batch normalization, introduced by Ioffe and Szegedy in 2015, fundamentally alters how neural networks are trained by stabilizing the distribution of layer inputs, specifically addressing the challenge of internal covariate shift. I've consistently observed that models trained without batch normalization tend to require more careful initialization, lower learning rates, and are generally less robust to hyperparameter changes compared to their counterparts incorporating this technique. In Keras, batch normalization is implemented as a dedicated layer, facilitating straightforward integration into diverse architectures.

Essentially, batch normalization operates on each mini-batch during training. It normalizes the activations of a specific layer, making them have zero mean and unit variance. Subsequently, the normalized values are scaled and shifted by learned parameters. This dual operation—normalization followed by a linear transformation—allows the network to learn the optimal distribution for each layer's inputs, which isn't rigidly fixed to a unit normal.

The primary mechanism involves calculating the mean and variance for each activation feature (across the batch). Let’s denote the mini-batch of layer inputs as *B*, where *B* = {*x*<sub>1</sub>, ..., *x<sub>m</sub>*} with *m* samples. First, the mini-batch mean (*μ*<sub>B</sub>) is calculated:

   *μ*<sub>B</sub> = (1/*m*) ∑<sub>i=1</sub><sup>*m*</sup> *x*<sub>i</sub>

Next, the mini-batch variance (*σ*<sub>B</sub><sup>2</sup>) is calculated:

   *σ*<sub>B</sub><sup>2</sup> = (1/*m*) ∑<sub>i=1</sub><sup>*m*</sup> (*x*<sub>i</sub> - *μ*<sub>B</sub>)<sup>2</sup>

These are used to normalize each activation:

   *x̂*<sub>i</sub> = (*x*<sub>i</sub> - *μ*<sub>B</sub>) / √(*σ*<sub>B</sub><sup>2</sup> + *ε*)

Here, *ε* is a small constant (typically around 1e-5) added for numerical stability, preventing division by zero if the variance is close to zero. Finally, the normalized values are scaled and shifted:

   *y*<sub>i</sub> = *γ* *x̂*<sub>i</sub> + *β*

*γ* (scale) and *β* (shift) are learnable parameters, learned during training through backpropagation. These parameters enable the network to recover the original activation distribution if it proves advantageous or learn a more effective one. During inference, rather than using batch statistics, running (or population) means and variances computed during training are used for normalization, which ensures consistent performance. This is an important distinction from the training phase where statistics are calculated on the fly.

Here’s how this manifests in Keras, through several examples.

**Example 1: Batch Normalization after a Dense Layer**

```python
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])
```

In this example, batch normalization is applied after a dense (fully connected) layer. The `BatchNormalization` layer will normalize the activations that come out of the `Dense(64)` layer. It's crucial to understand that batch norm often (but not always) occurs *before* an activation function is applied, although the choice can depend on the specific network architecture and performance. In this example, a ReLU activation is applied prior to normalization, though it’s not a strict rule. The input shape `input_shape=(784,)` specifies the input dimension of the network. The `Dense(10, activation='softmax')` layer represents the output layer for a classification task, with each of the 10 neurons corresponding to a class.

**Example 2: Batch Normalization within a Convolutional Neural Network (CNN)**

```python
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

```

This illustrates using batch normalization within a CNN, a common application of the technique. Notice that `BatchNormalization` is inserted after each convolutional layer (`Conv2D`). This means the normalization will happen feature-map wise, across the batch. The input shape for the CNN is `(28, 28, 1)`, representing a grayscale image. `MaxPooling2D` layers are used for downsampling. `Flatten` reshapes the output from the convolutional layers into a one-dimensional vector prior to feeding into the fully connected output layer.

**Example 3: Customizing Batch Normalization Parameters**

```python
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(momentum=0.99, epsilon=0.001, beta_initializer='zeros', gamma_initializer='ones'),
    layers.Dense(10, activation='softmax')
])
```

In this example, I demonstrate how you can customize the parameters of the `BatchNormalization` layer. The `momentum` parameter controls the contribution of past batches when calculating the running mean and variance. Larger momentum values prioritize more recent batches. The `epsilon` parameter adds a small value for numerical stability during division. Furthermore, custom initialization for the scale `gamma` and shift `beta` parameters is specified here, defaulting to ones and zeros respectively. This level of customization can sometimes improve model training.

Several resources offer further detailed understanding. I would advise reading the original batch normalization paper by Ioffe and Szegedy for a detailed technical description. Additionally, various deep learning textbooks provide comprehensive discussions of the topic. Online courses that focus on deep learning often include sections on batch normalization that delve into its practical aspects. Finally, exploring different variations of batch normalization, such as layer normalization or group normalization, may prove useful in broadening the understanding of these normalization techniques. Experimentation is key to determine optimal usage and placement within a given architecture.

Through my own experiences in training models, I have found the use of batch normalization to be practically essential, especially when dealing with deeper networks. It allows for faster convergence and higher robustness, thus making the entire model training process considerably more reliable. While specific implementation details can be intricate, understanding the underlying principles of mean-variance normalization with learnable parameters is essential for its effective utilization within diverse architectures.
