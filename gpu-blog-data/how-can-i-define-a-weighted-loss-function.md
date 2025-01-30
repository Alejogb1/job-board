---
title: "How can I define a weighted loss function for a Keras CNN image classifier in TensorFlow 2.0+?"
date: "2025-01-30"
id: "how-can-i-define-a-weighted-loss-function"
---
Class imbalance is a prevalent challenge in image classification tasks, and directly impacts model performance. Using standard loss functions like categorical cross-entropy often yields suboptimal results when dealing with datasets where certain classes significantly outnumber others. Therefore, incorporating class weights into the loss calculation is essential to prioritize learning from minority classes, ultimately leading to a more balanced and robust classifier.

**Explanation of Weighted Loss**

The fundamental idea behind weighted loss is to adjust the contribution of each training sample to the overall loss computation based on its class membership. Instead of treating all samples equally, we introduce weights that amplify the importance of samples from underrepresented classes and, conversely, reduce the influence of samples from overrepresented ones. This helps prevent the model from gravitating toward the majority class, which would yield high overall accuracy but poor performance on minority classes.

Let’s consider a scenario I encountered while building a convolutional neural network to classify images of various types of plants. My initial dataset had significantly more images of common plants, such as dandelions, compared to rarer ones, like orchids. The classifier trained with standard categorical cross-entropy consistently performed poorly on orchids, even after prolonged training. This imbalance resulted in a model that primarily optimized for correctly identifying dandelions, thus demonstrating the necessity of a weighted loss function.

The core concept revolves around scaling the loss value computed for each sample. Suppose we have a categorical classification problem with *C* classes. In typical cross-entropy loss, the loss for a single sample (often represented as a one-hot vector *y_true*) and the model's prediction (*y_pred*) can be expressed. With weighted loss, we introduce a class-specific weight, *w_c*, corresponding to the correct class label, *c*. The loss component for each sample now involves multiplying the standard cross-entropy term by this corresponding weight *w_c*.

Calculating these weights is critical. A common strategy is to use inverse class frequency. The weight, *w_c*, for class *c* is usually calculated as *n / (C* *n_c*)*, where *n* is the total number of training samples, *n_c* is the number of samples in class *c*, and *C* is the total number of classes. This ensures that classes with fewer samples receive higher weights and, as a consequence, influence the training more significantly. Alternatively, other weighting strategies exist that may suit specific situations, for example, penalizing classes more aggressively with inverse class frequency squared.

In TensorFlow, the Keras API allows the usage of sample weights either through `model.fit()` directly or by defining a custom loss function. Both offer flexibility, although using a custom function allows more control over the loss implementation and can handle more complex weighting strategies.

**Code Examples and Commentary**

**Example 1: Using `class_weight` parameter in `model.fit()`**

This approach offers a straightforward implementation for basic inverse class frequency weighting.

```python
import tensorflow as tf
import numpy as np

# Assume labels are one-hot encoded
y_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]]) # Example labels
y_train_indices = np.argmax(y_train, axis=1)
num_classes = y_train.shape[1]
n_samples = len(y_train)

# Calculate class frequencies
class_counts = np.bincount(y_train_indices, minlength=num_classes)
class_weights = n_samples / (num_classes * class_counts)

class_weight_dict = dict(enumerate(class_weights))

# Example of a simple CNN (placeholder, training is not performed)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training with class weights
model.fit(np.random.rand(len(y_train), 28, 28, 3), y_train, epochs=10, class_weight=class_weight_dict)
```

This code snippet calculates the inverse class frequency and then feeds the weights into the `model.fit` method as `class_weight`. The `class_weight` parameter expects a dictionary with class indices as keys and their corresponding weights as values. This is the simplest way to apply class weighting in many cases.

**Example 2: Custom Weighted Loss Function using TensorFlow API**

This demonstrates a custom weighted categorical cross-entropy loss, offering greater control over the weighting process.

```python
import tensorflow as tf
import numpy as np

# Assume labels are one-hot encoded
y_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]]) # Example labels
y_train_indices = np.argmax(y_train, axis=1)
num_classes = y_train.shape[1]
n_samples = len(y_train)

# Calculate class frequencies
class_counts = np.bincount(y_train_indices, minlength=num_classes)
class_weights = n_samples / (num_classes * class_counts)

# Custom loss function
def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true_indices = tf.argmax(y_true, axis=-1)
        weight = tf.gather(weights, y_true_indices)
        loss_value = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return loss_value * weight
    return loss

# Example of a simple CNN (placeholder, training is not performed)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss=weighted_categorical_crossentropy(class_weights), metrics=['accuracy'])

# Model training with class weights
model.fit(np.random.rand(len(y_train), 28, 28, 3), y_train, epochs=10)
```

In this example, I define `weighted_categorical_crossentropy` as a function that takes class weights and returns a callable loss function. Inside, the true class labels are used to retrieve the correct weights, and the standard categorical cross-entropy is then scaled by these weights. This offers complete control over the loss calculations.

**Example 3:  Sample Weighting during training using NumPy arrays**

This approach utilizes NumPy arrays to implement per-sample weights for each training instance.

```python
import tensorflow as tf
import numpy as np

# Assume labels are one-hot encoded
y_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]]) # Example labels
y_train_indices = np.argmax(y_train, axis=1)
num_classes = y_train.shape[1]
n_samples = len(y_train)

# Calculate class frequencies
class_counts = np.bincount(y_train_indices, minlength=num_classes)
class_weights = n_samples / (num_classes * class_counts)

# Prepare sample weights
sample_weights = class_weights[y_train_indices]

# Example of a simple CNN (placeholder, training is not performed)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training with sample weights
model.fit(np.random.rand(len(y_train), 28, 28, 3), y_train, epochs=10, sample_weight=sample_weights)
```

In this case, a `sample_weights` array of length equal to the number of samples was created. Each element in this array represents the class weight for the corresponding sample. The array is passed directly to the `fit()` method via the `sample_weight` parameter.

**Resource Recommendations**

For detailed understanding of loss functions, I recommend exploring resources related to TensorFlow documentation and numerical optimization principles. Textbooks on deep learning can provide further background on the mathematical basis for these methods. Research papers related to handling imbalanced datasets can offer further insight into advanced weighting methods.
