---
title: "Which approach, a single sigmoid output or two linear outputs, performs better for binary classification using TensorFlow?"
date: "2025-01-30"
id: "which-approach-a-single-sigmoid-output-or-two"
---
Implementing binary classification models, I've repeatedly observed that the choice between a single sigmoid output and two linear outputs, followed by a softmax, significantly impacts model behavior and ease of optimization. The apparent equivalence on paper often masks practical nuances encountered during training. A core distinction lies not solely in their output range constraints (sigmoid limiting outputs to [0,1], whereas softmax normalizes across outputs), but also in how each interacts with commonly used loss functions.

The single sigmoid output paired with binary cross-entropy loss is a direct implementation of logistic regression principles, inherently suited for scenarios where the model is predicting a probability of a single, positive class. Conversely, two linear outputs followed by softmax, combined with categorical cross-entropy loss, models the classification process as a choice between two distinct categories, yielding class probabilities. While both approaches ultimately aim to discern a binary decision boundary, their training dynamics and interpretability can differ considerably.

A single sigmoid output, specifically, directly models the probability of the positive class. The value ranges between 0 and 1, representing the likelihood of that class. When calculating the binary cross-entropy loss, only one target value per sample is used: either 0 (negative class) or 1 (positive class). During backpropagation, adjustments are applied to minimize the error between this single sigmoid output and the one-hot representation of the class target. This setup is computationally less intensive, as it avoids the additional softmax operation and has fewer output nodes. Furthermore, the single sigmoid output provides a very direct interpretation: values closer to 1 imply higher confidence of a sample belonging to the positive class, and vice versa.

The alternative, using two linear outputs followed by a softmax, maps input features to two distinct values before normalizing them into a probability distribution over the two classes. This approach utilizes categorical cross-entropy. This loss function expects a one-hot encoded vector of the ground truth (e.g., [1, 0] for the negative class, [0, 1] for the positive class). The softmax then calculates probabilities of the input belonging to each class. During backpropagation, gradient updates are determined by comparing these probabilities to the one-hot encoded target, adjusting weights to minimize the discrepancy. The softmax step ensures the two outputs sum to one, providing an inter-class probability view. While this setup may seem redundant for binary classification, it’s the standard practice when extending classification problems to multiple classes, and may offer a slightly different training signal due to its different gradient characteristics.

Consider a scenario using the TensorFlow Keras API. I will demonstrate both setups.

**Example 1: Single Sigmoid Output**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Model definition with single sigmoid output
def build_sigmoid_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Sigmoid activation for binary
    ])
    return model

# Example Usage
input_shape = (10,) # Example input shape with 10 features
model_sigmoid = build_sigmoid_model(input_shape)
model_sigmoid.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Example input
sample_input = tf.random.normal((1, 10))
prediction_sigmoid = model_sigmoid(sample_input)
print(f"Sigmoid output probability: {prediction_sigmoid.numpy()}")
```

This code defines a model that takes an input of shape `(10,)` through two hidden layers, then culminates with a single output node using a sigmoid activation. The `binary_crossentropy` loss function is explicitly chosen for its suitability with the sigmoid's [0, 1] range of output. The predicted output provides a probability of belonging to the positive class.

**Example 2: Two Linear Outputs with Softmax**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Model definition with two linear outputs and softmax
def build_softmax_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)  # 2 linear outputs
    ])
    return model

# Example Usage
input_shape = (10,) # Example input shape with 10 features
model_softmax = build_softmax_model(input_shape)
model_softmax.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Example input
sample_input = tf.random.normal((1, 10))
# Get raw linear outputs
raw_outputs = model_softmax(sample_input)
# Apply softmax activation
probabilities = tf.nn.softmax(raw_outputs)

print(f"Softmax output probabilities: {probabilities.numpy()}")
```

Here, the output layer consists of two linear nodes. Unlike Example 1, we do not directly use an activation function within the model definition (as the raw linear outputs are processed by softmax outside). During training (not included in the example), we use `categorical_crossentropy` loss coupled with one-hot encoding of target variables. When applying inference (as shown), we utilize `tf.nn.softmax` to generate a probability distribution over the two classes.

**Example 3: Training and comparison**

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_sigmoid_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_softmax_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)
    ])
    return model

# Example Usage - Training with some sample data
input_shape = (10,)
num_samples = 1000
X_train = np.random.rand(num_samples, *input_shape)
y_train_sigmoid = np.random.randint(0, 2, size=(num_samples, 1)) # single target for sigmoid model
y_train_softmax = np.eye(2)[np.random.randint(0, 2, size=num_samples)] # one-hot encoded for softmax

# Model 1: Training with Sigmoid
model_sigmoid = build_sigmoid_model(input_shape)
model_sigmoid.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_sigmoid = model_sigmoid.fit(X_train, y_train_sigmoid, epochs=10, verbose=0)

# Model 2: Training with Softmax
model_softmax = build_softmax_model(input_shape)
model_softmax.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_softmax = model_softmax.fit(X_train, y_train_softmax, epochs=10, verbose=0)

# Comparing final training metrics
print(f"Sigmoid model training accuracy: {history_sigmoid.history['accuracy'][-1]}")
print(f"Softmax model training accuracy: {history_softmax.history['accuracy'][-1]}")
```

This final example shows training data being generated, then passed to both versions of the model. The accuracy at the end of training is printed to show a basic comparison. In practice, the optimal architecture would depend on the exact dataset and objective, so these comparisons are limited.

In terms of general practice, both are valid approaches. For purely binary classification, the sigmoid with binary cross-entropy is computationally more efficient and straightforward, making it my default starting point. The softmax with categorical cross-entropy, however, provides a more direct avenue for later expansion to multi-class problems, as well as offering a probability distribution over all possible classes that can be beneficial in certain analytical contexts. Moreover, I have observed that the softmax formulation can sometimes exhibit slightly more stability or quicker convergence in highly complex binary classification tasks. However, there's no universal winner; often, I test both to determine the ideal fit for a given project.

When choosing an approach, practical considerations are key. For straightforward binary tasks, I lean towards the single sigmoid due to its directness. Conversely, if the project might evolve to include multi-class classification later, or the need for a formal probability distribution across two classes emerges, I prefer the two linear output softmax method, albeit with the added implementation overhead.

For further learning, I highly recommend investigating the theoretical basis behind logistic regression and cross-entropy loss, as well as the mathematical properties of softmax. Textbooks and online courses focused on machine learning are a good starting point. The official TensorFlow documentation provides detailed explanations about Keras layers and loss functions which is also an invaluable resource. Studying the inner workings of gradient descent algorithms and how they are affected by different output representations is also crucial for a complete grasp of the behavior of the model. Finally, practical experimentation on your own is often the best teacher.
