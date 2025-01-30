---
title: "How can TensorFlow batches be manually selected during deep learning training?"
date: "2025-01-30"
id: "how-can-tensorflow-batches-be-manually-selected-during"
---
TensorFlow's default training pipeline processes data in sequential batches, but situations often demand finer control over batch selection for tasks such as curriculum learning, adversarial training, or handling imbalanced datasets. Directly manipulating the batch selection process involves going beyond the standard `tf.data.Dataset` iteration provided by methods like `model.fit()`. I've encountered scenarios, particularly when dealing with highly sparse data in anomaly detection, where the conventional approach proved inadequate, necessitating custom batch selection strategies.

The core issue stems from `tf.data.Dataset`'s design, which primarily focuses on efficient data pipelining and preprocessing. While the `Dataset` API allows for transformations like shuffling and batching, it doesn't inherently support direct indexing or conditional batch selection during training. To achieve this, the dataset needs to be treated more as a source, and batch construction must occur outside the default iteration loop. I typically approach this by implementing a custom training loop leveraging the `tf.data.Dataset`’s `as_numpy_iterator()` or by creating custom data loaders.

The typical training flow using `model.fit` abstracts away the batch creation.  For instance:

```python
import tensorflow as tf
import numpy as np

# Sample Data Creation
X = np.random.rand(100, 20)
y = np.random.randint(0, 2, 100)
dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=5)
```

This code illustrates standard training, where the batches are formed automatically and sequentially by TensorFlow. We will modify this type of procedure.

**Example 1: Random Batch Selection**

My initial need for this approach arose when implementing a custom curriculum learning strategy. The goal was to gradually expose the model to more complex data patterns. This involved selecting training batches based on a dynamically changing difficulty score assigned to each data point. To do this, we must move away from the built-in `model.fit()` and implement our own training loop. Here's how I approached random selection:

```python
import tensorflow as tf
import numpy as np
import random

# Sample Data Creation
X = np.random.rand(100, 20)
y = np.random.randint(0, 2, 100)
dataset = tf.data.Dataset.from_tensor_slices((X, y)).cache().as_numpy_iterator()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()
batch_size = 32
num_epochs = 5

for epoch in range(num_epochs):
    all_indices = list(range(100))
    random.shuffle(all_indices)

    for i in range(0, 100, batch_size):
      batch_indices = all_indices[i:i+batch_size]
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]
      with tf.GradientTape() as tape:
        predictions = model(X_batch)
        loss = loss_fn(y_batch, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f"Epoch: {epoch+1}")
```

This example first converts the `tf.data.Dataset` to a numpy iterator with `as_numpy_iterator()` for more granular control. It then creates a list of all sample indices, shuffles them randomly, and then steps through the list to manually create batches. The important point here is that batch selection is now explicitly within our control. The dataset is cached for efficiency as this iterator will be accessed many times.

**Example 2: Priority-Based Batch Selection**

A subsequent project involved training a model on an imbalanced dataset where certain classes were significantly underrepresented. To mitigate this, I implemented a priority-based sampling scheme, selecting batches with higher representation of the minority class in each epoch. I achieved this by calculating class weights and using these weights to bias the random selection of batches. This requires having class information available, which can be done by either creating the indices as above or creating a modified `tf.data.Dataset` object as below.

```python
import tensorflow as tf
import numpy as np
import random

# Sample Data Creation (Imbalanced)
X = np.random.rand(100, 20)
y = np.concatenate([np.zeros(80, dtype=int), np.ones(20, dtype=int)]) # Imbalanced classes
dataset = tf.data.Dataset.from_tensor_slices((X, y)).cache().as_numpy_iterator()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()
batch_size = 32
num_epochs = 5

# Calculate Class Weights
class_counts = np.bincount(y)
class_weights = 1.0 / (class_counts + 1e-6) #Adding small number to avoid dividing by zero.
sample_weights = class_weights[y]
# Normalize Weights to sum to 1.
sample_weights = sample_weights / np.sum(sample_weights)

for epoch in range(num_epochs):
    all_indices = list(range(100))
    batch_indices = np.random.choice(all_indices, size=100, replace=False, p=sample_weights) #Weighted Random Selection

    for i in range(0, 100, batch_size):
      batch_indices_subset = batch_indices[i:i+batch_size]
      X_batch = X[batch_indices_subset]
      y_batch = y[batch_indices_subset]
      with tf.GradientTape() as tape:
        predictions = model(X_batch)
        loss = loss_fn(y_batch, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f"Epoch: {epoch+1}")

```

This example builds upon the random batch selection concept but introduces weighted selection using `np.random.choice` and sample weights which are calculated based on the class imbalance. The weights are passed as probability of selection to the `np.random.choice`. By calculating and using these weights, the model is exposed to the minority class more frequently, leading to better generalization.

**Example 3: Custom Data Loader Class**

For complex batching logic, especially when multiple data sources are involved, I often find it beneficial to encapsulate the logic into a custom data loader class.  This makes the code cleaner, more reusable, and easier to maintain, as well as more flexible.

```python
import tensorflow as tf
import numpy as np
import random

class CustomDataLoader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = X.shape[0]

    def get_batch(self, indices):
      X_batch = self.X[indices]
      y_batch = self.y[indices]
      return X_batch, y_batch

    def __iter__(self):
        self.all_indices = list(range(self.num_samples))
        return self

    def __next__(self):
      if not self.all_indices:
        raise StopIteration
      batch_indices = random.sample(self.all_indices, min(self.batch_size,len(self.all_indices)))
      self.all_indices = [i for i in self.all_indices if i not in batch_indices]
      X_batch, y_batch = self.get_batch(batch_indices)
      return X_batch, y_batch

# Sample Data Creation
X = np.random.rand(100, 20)
y = np.random.randint(0, 2, 100)

data_loader = CustomDataLoader(X, y, batch_size=32)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()
num_epochs = 5

for epoch in range(num_epochs):
  for X_batch, y_batch in data_loader:
      with tf.GradientTape() as tape:
          predictions = model(X_batch)
          loss = loss_fn(y_batch, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  print(f"Epoch: {epoch+1}")
```

This example presents a `CustomDataLoader` class which manages batch generation. It implements the iterator protocol (`__iter__` and `__next__`) which means it can be directly used in a `for` loop. This greatly simplifies the training loop, making it more readable and adaptable. This method is beneficial when batch selection logic becomes too intricate for simple scripts.

In essence, achieving manual batch selection during TensorFlow training involves moving away from the high-level abstraction of `model.fit()` and implementing a custom training loop that either iterates over the data or via a custom iterator class. The examples above highlight methods to achieve this, ranging from simple random selection to more complex priority-based selection, and a custom data loader class.  For further exploration into effective data handling during model training, I recommend looking at material on advanced data loading techniques for deep learning, particularly focusing on balanced data sampling, curriculum learning, and advanced data pipelining techniques. Resources that discuss custom training loops in TensorFlow would also be helpful. Understanding how `tf.GradientTape` operates is also key to implementing successful custom training loops.
