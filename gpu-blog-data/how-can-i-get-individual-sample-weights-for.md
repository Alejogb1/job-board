---
title: "How can I get individual sample weights for each class in Keras?"
date: "2025-01-30"
id: "how-can-i-get-individual-sample-weights-for"
---
The computation of individual sample weights, particularly when aiming for class-balanced learning in Keras, requires a departure from the typical single `sample_weight` input. Keras' `fit` method primarily accepts a single array of weights corresponding to each training instance. To achieve per-class weighting, a preprocessing step is necessary to expand these class-based weights into sample-level weights. I've encountered this challenge multiple times, especially when dealing with highly imbalanced datasets in medical imaging analysis, where certain pathologies are significantly rarer than others. The direct approach is to leverage the class labels to construct an array that matches the dimensions of the training data. This array then acts as the sample weight during training.

**Explanation**

Keras' `fit` function accepts a `sample_weight` argument, which should be a 1D array with the same number of elements as the training data. This is the key constraint. The objective, when desiring per-class weights, is to map class-level weights to their respective instances within the dataset. Imagine a binary classification problem with two classes, 0 and 1. You may want to assign a weight of 1 to all instances of class 0 and a weight of 5 to all instances of class 1 due to imbalance. We cannot directly pass `[1, 5]` to the `sample_weight` argument. Instead, we need an array of the same length as the input data where each element corresponds to either 1 or 5 based on the class of the data point.

This methodology involves two core steps:
1. **Class Weight Calculation**: First, compute the class weights based on the inverse frequency, median frequency or any other strategy you choose. These are typically represented as a dictionary or another map structure where keys are the class labels and values are the associated class weights.
2. **Sample Weight Generation**: Second, create a 1D NumPy array (or its equivalent in your numerical library) that has the same length as your training data. Iterate through your training labels and, for each data point, assign it the class weight corresponding to its label. This array becomes the `sample_weight` array that will be passed to the Keras `fit` method.

Crucially, this process must occur *before* the model training, allowing the Keras optimizer to use the per-sample weight during the loss calculation. It's important to note that this is a data preprocessing step, not an intrinsic part of Keras' training loop. Any modifications to class weight calculations or application should be reflected by regenerating this `sample_weight` array. Additionally, this approach can be extended to multi-class problems, the number of weights will simply be equal to the number of unique classes.

**Code Examples**

Here are three examples illustrating this process, incorporating varying complexities:

**Example 1: Binary classification with simple inverse frequency weights**

```python
import numpy as np
import tensorflow as tf

# Fictional dataset and labels
y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]) # Imbalanced classes
X_train = np.random.rand(len(y_train), 10)  # 10 features per data point

# Calculate class weights (inverse frequency)
classes, counts = np.unique(y_train, return_counts=True)
class_weights = {c: len(y_train) / (count * len(classes)) for c, count in zip(classes, counts)}
print("Class Weights:", class_weights)

# Generate sample weights
sample_weights = np.array([class_weights[label] for label in y_train])
print("Sample Weights:", sample_weights)

# Build a simple model for demonstration
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=4, sample_weight=sample_weights)
```

*Commentary:*  This first example demonstrates the basic principle. The class weights are computed based on the inverse frequency of samples in each class. I prefer this method when dataset imbalances are relatively moderate. The resulting `sample_weights` array corresponds to the training labels. These weights are then passed to the Keras `fit` function during model training. Note that while inverse frequency is easy to calculate, it may be too aggressive for significant class imbalances.

**Example 2: Multi-class classification with adjusted class weights**

```python
import numpy as np
import tensorflow as tf

# Fictional multi-class dataset
y_train = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]) # Imbalanced classes
X_train = np.random.rand(len(y_train), 10)

# Calculate class weights (adjusted inverse frequency)
classes, counts = np.unique(y_train, return_counts=True)
max_count = np.max(counts)
class_weights = {c: max_count / count for c, count in zip(classes, counts)}
print("Class Weights:", class_weights)


# Generate sample weights
sample_weights = np.array([class_weights[label] for label in y_train])
print("Sample Weights:", sample_weights)


# Build a simple model for demonstration
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 output classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=4, sample_weight=sample_weights)
```

*Commentary:*  This example extends the principle to multi-class problems. Here, instead of a binary classification, we have three classes. The class weights are calculated in a way that emphasizes the less frequent classes without excessively downplaying the more frequent ones; each weight is relative to the maximum frequency. It uses `sparse_categorical_crossentropy` for the loss, aligning with multi-class label encoding. Note the `softmax` activation on the final layer, standard for multi-class problems.  The core idea, mapping class weights to per-sample weights, remains consistent. I find this method to perform better in scenarios with more than two classes, as it prevents too much dominance of under-represented classes.

**Example 3: Using a pre-computed weight map and a custom mapping strategy**

```python
import numpy as np
import tensorflow as tf

# Fictional dataset with labels encoded as a string ID (simulating a common use case)
y_train_ids = ["a1", "a1", "a2", "a2", "a2", "b1", "b1", "b1", "b1", "b1", "b2", "b2", "b2"]
X_train = np.random.rand(len(y_train_ids), 10)

# Pre-computed class weights, common in research scenarios
class_weights_map = { "a1": 1.5, "a2": 3.2, "b1": 0.5, "b2": 1.0 }

# Generate sample weights
sample_weights = np.array([class_weights_map[label] for label in y_train_ids])
print("Sample Weights:", sample_weights)

# Convert label IDs to numerical values for keras compatibility
unique_ids = np.unique(y_train_ids)
label_mapping = {label: i for i, label in enumerate(unique_ids)}
y_train = np.array([label_mapping[label_id] for label_id in y_train_ids])

# Build a simple model for demonstration
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(len(unique_ids), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=4, sample_weight=sample_weights)
```

*Commentary:* This example demonstrates a scenario where the class weights are pre-determined, perhaps through experimentation or domain knowledge, and also how label values might need to be encoded if they are not integers. The `class_weights_map` dictionary is not derived from data frequencies. This simulates a situation where we have specialized weights we'd like to test.  This is very common in medical or physics analysis. The important point is that this methodology directly accommodates arbitrary class-based weights as long as we can map them to individual training instances using `y_train_ids` in this case and then use them when training using the `sample_weights` argument.

**Resource Recommendations**

While the Keras API documentation provides details on the `fit` method, resources specifically on imbalanced learning often prove helpful. Look for literature and tutorials focusing on handling imbalanced datasets in machine learning.  Publications discussing techniques like focal loss, cost-sensitive learning, and other re-weighting techniques frequently offer deeper insights into appropriate methods for sample weighting. Additionally, exploring libraries and frameworks beyond core Keras, such as those dedicated to data analysis and management, can offer practical methods for preparing the data prior to training. It is useful to examine the implementations of loss functions in the Keras source to better understand how `sample_weight` is used when calculating gradients.
