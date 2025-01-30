---
title: "How does logistic regression using TensorFlow 2.1.0 perform on the Iris dataset?"
date: "2025-01-30"
id: "how-does-logistic-regression-using-tensorflow-210-perform"
---
Logistic regression, despite its simplicity, serves as a foundational classification algorithm. I've observed its behavior extensively across numerous datasets, and the Iris dataset provides a particularly illuminating case study, specifically when implemented with TensorFlow 2.1.0, a version that offers both eager execution for debugging and graph compilation for performance.

My practical experience suggests that logistic regression's performance on Iris is predictably solid due to several factors intrinsic to the dataset: the linear separability of most classes and the relatively low dimensionality. However, performance isn’t just about accuracy; it encompasses training speed, resource utilization, and the ease of interpretation. Using TensorFlow allows for granular control over these aspects.

Here's how logistic regression is applied to the Iris dataset within a TensorFlow 2.1.0 context: the Iris dataset, consisting of 150 samples described by four features (sepal length, sepal width, petal length, and petal width) and three classes (setosa, versicolor, and virginica), represents a multiclass classification problem. Logistic regression, being inherently a binary classifier, necessitates a transformation for multiclass scenarios. Typically, this is achieved using the one-vs-rest approach (also called one-vs-all). For each class, a separate logistic regression model is trained, distinguishing that class against all others. The model's output for a given input is the class with the highest predicted probability.

TensorFlow manages the underlying mathematics. The core logistic regression calculation involves the sigmoid function applied to the weighted sum of input features, ultimately providing a probability between 0 and 1. The weights are learned during the training process, minimized based on a loss function, usually cross-entropy. For multi-class classification, the softmax function extends the sigmoid.

**Code Example 1: Basic Logistic Regression Implementation**

This snippet demonstrates a direct implementation using TensorFlow. It focuses on clarity rather than advanced techniques.

```python
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert y to one-hot encoding
y_one_hot = tf.one_hot(y, depth=3)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot.numpy(), test_size=0.2, random_state=42)

# Define model parameters
learning_rate = 0.01
epochs = 1000
input_size = X_train.shape[1]  # 4 features
output_size = 3             # 3 classes

# Initialize weights and bias
W = tf.Variable(tf.random.normal(shape=(input_size, output_size)))
b = tf.Variable(tf.zeros(shape=(output_size,)))


# Define the model and loss function
def logistic_regression(x):
    logits = tf.matmul(x, W) + b
    return logits

def loss_fn(y_true, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))

# Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Train the model
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = logistic_regression(X_train)
        loss = loss_fn(y_train, logits)
    grads = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.numpy():.4f}")

# Model Evaluation
logits_test = logistic_regression(X_test)
predictions = tf.argmax(logits_test, axis=1)
true_labels = tf.argmax(y_test, axis=1)

accuracy = np.mean(predictions.numpy() == true_labels.numpy())
print(f"\nTest Accuracy: {accuracy:.4f}")

```

In this first example, a basic gradient descent is implemented from scratch. We define weights `W` and bias `b`, the forward pass `logistic_regression` using matrix multiplication for efficiency and a cross-entropy loss function. Training proceeds in a loop, utilizing TensorFlow's `GradientTape` to compute gradients. It’s a fairly verbose and manually coded example that provides an understanding of what’s going on under the hood. The final section provides a quick accuracy score over the test set.

**Code Example 2: Using `tf.keras.layers` for Clarity**

This snippet leverages the higher-level API of Keras, a part of TensorFlow. It prioritizes brevity and encapsulation.

```python
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert y to one-hot encoding
y_one_hot = tf.one_hot(y, depth=3)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot.numpy(), test_size=0.2, random_state=42)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', input_shape=(4,))
])

# Compile the model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
```

In this second example, the code is significantly more concise. The entire logistic regression model is encapsulated within the Keras `Sequential` model which is a higher level abstraction. We define a single dense layer which performs both the weighted sum of inputs and the application of the softmax function. The `model.compile` method configures the optimizer, loss function and evaluation metrics and the `model.fit` method does all the parameter updates in the background. The model evaluation is greatly simplified with the `model.evaluate` method. This highlights the ease of use offered by high-level abstractions.

**Code Example 3: Incorporating Batching and Validation**

This final example extends the previous one by integrating mini-batching and validation, which I found crucial for stabilizing the training process on larger datasets.

```python
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert y to one-hot encoding
y_one_hot = tf.one_hot(y, depth=3)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot.numpy(), test_size=0.2, random_state=42)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', input_shape=(4,))
])

# Compile the model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create TensorFlow Datasets for batching
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)


# Train the model
model.fit(train_dataset, epochs=1000, validation_data=test_dataset, verbose=0)


# Evaluate the model
loss, accuracy = model.evaluate(test_dataset, verbose=0)

print(f"Test Accuracy: {accuracy:.4f}")
```

This third example integrates the use of TensorFlow datasets to perform batching, which is a crucial step to take in any deep learning project. The dataset is wrapped within the `tf.data.Dataset.from_tensor_slices` and then batched using the `.batch` method. This allows the training loop to now loop over batches rather than the whole dataset at once. Also, validation data is also passed to the training loop allowing the validation loss to be calculated during training for monitoring and model selection.

Regarding performance on the Iris dataset, accuracy typically exceeds 90% when correctly trained using these methods. The primary benefit of using TensorFlow is its flexibility in model design and implementation. One can easily scale these models to large datasets and easily port them across CPU/GPU. The eager execution mode also facilitates debugging.

For a deeper dive into logistic regression and TensorFlow, I recommend exploring the TensorFlow documentation itself, particularly the sections on Keras layers and optimizers. Furthermore, texts focusing on machine learning fundamentals, such as *Elements of Statistical Learning* and *Pattern Recognition and Machine Learning*, would be beneficial. The scikit-learn library documentation also offers insights into the mathematics behind logistic regression. In my experience, combining theoretical knowledge with practical experimentation is essential to fully understand and utilize these tools.
