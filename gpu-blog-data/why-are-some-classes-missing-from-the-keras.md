---
title: "Why are some classes missing from the Keras deep neural network?"
date: "2025-01-30"
id: "why-are-some-classes-missing-from-the-keras"
---
The apparent absence of certain classes within a Keras model, particularly after training, is frequently not a case of them being "missing" in the sense of being deleted or excluded programmatically. Instead, it typically indicates a lack of sufficient representation during the training process, manifesting as zero or near-zero predicted probabilities for those underrepresented classes during inference. I've encountered this issue numerous times, particularly when working with highly imbalanced datasets where some classes have significantly fewer training samples than others.

Fundamentally, a Keras model, like most neural networks, learns by adjusting its internal weights and biases to minimize a defined loss function across the provided training data. If a specific class is significantly underrepresented, the gradient updates driving learning will be biased towards the prevalent classes. This often leads to the model effectively ignoring or failing to discern patterns associated with the minority class, resulting in consistently low predicted probabilities regardless of the input. The classes aren’t missing from the model's architecture; they're missing from its ability to accurately predict them. This situation is often exacerbated when the model is initialized with random weights, where the initial gradients do not point it towards even attempting to learn these rare features.

Let’s consider a practical scenario where we are classifying images of different types of fruit. We have a vast dataset of apples and oranges but only a handful of grape images. After training, it's very likely that the model will struggle, or even fail completely, to classify grapes, not because grapes are excluded, but because the training signal for that class was weak.

A critical factor affecting this behavior is the choice of loss function. Using a standard categorical cross-entropy loss with unbalanced class distribution will inherently push the model to focus on the classes that contribute most to the total loss, which will be the more frequent classes. The model might even learn to simply predict the majority class in every case, achieving high accuracy (as seen by the percentage correct) due to the class imbalance, while completely ignoring the minority classes. While accuracy can be misleading in this setting, the model output will reflect that. Even when training data is present for a less represented class, a model may still exhibit underperformance if it is trained alongside classes where training signal is overwhelming.

Now, let's delve into some code examples demonstrating these issues and methods to address them. We’ll be using a simple convolutional model for image classification.

**Example 1: Simple Model with Unbalanced Data**

This example shows the problem without taking steps to mitigate class imbalance. We create synthetic, unbalanced datasets to simulate a real-world condition.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Create synthetic, unbalanced dataset
def create_data(num_samples_per_class):
    X = []
    y = []
    num_classes = 3
    for label in range(num_classes):
       for _ in range(num_samples_per_class[label]):
           img = np.random.rand(28, 28, 3) # Simple 28x28x3 image
           X.append(img)
           y.append(label)
    return np.array(X), np.array(y)

num_samples_per_class = [500, 400, 50] # Class imbalance
X, y = create_data(num_samples_per_class)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes=3)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax') # Output layer with 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=0)

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
actual_labels    = np.argmax(y_test, axis=1)

print(f"Predicted labels: {predicted_labels[0:50]}") #Shows bias toward class 0 and 1
print(f"Actual labels: {actual_labels[0:50]}")

```

In this example, even though the model has three output neurons, it is likely to demonstrate a high prediction probability for class 0 and class 1 while almost completely neglecting class 2.  The model is essentially ignoring the 'minority class' during the classification step, even though the class weights are not explicitly absent from the output layer. The output predictions will show the model almost always predicting class 0 or class 1.

**Example 2: Addressing Imbalance with Class Weights**

One of the most straightforward strategies to address imbalanced datasets is to adjust the class weights during the loss function computation.  This assigns more weight to the loss incurred from misclassifying samples from underrepresented classes, forcing the model to pay more attention to them.

```python
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Create synthetic, unbalanced dataset
def create_data(num_samples_per_class):
    X = []
    y = []
    num_classes = 3
    for label in range(num_classes):
       for _ in range(num_samples_per_class[label]):
           img = np.random.rand(28, 28, 3) # Simple 28x28x3 image
           X.append(img)
           y.append(label)
    return np.array(X), np.array(y)

num_samples_per_class = [500, 400, 50] # Class imbalance
X, y = create_data(num_samples_per_class)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights)) # Convert to a dictionary
print(f"Class weights: {class_weight_dict}")

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes=3)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax') # Output layer with 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=0, class_weight=class_weight_dict)

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
actual_labels    = np.argmax(y_test, axis=1)


print(f"Predicted labels: {predicted_labels[0:50]}") #Improved predictions on the third class
print(f"Actual labels: {actual_labels[0:50]}")
```
By implementing class weights, we shift the model's learning towards the underrepresented classes. This is reflected in the output as the model now exhibits a greater likelihood of correctly classifying samples from all classes and is no longer biased toward the majority classes. The `sklearn` library provides utility functions to calculate balanced weights.

**Example 3: Data Augmentation**

Another approach is to augment the existing data for underrepresented classes. This can artificially increase the number of training samples and help the model generalize better. I demonstrate this using a simplified augmentation strategy, as a full image augmentation strategy can be complex.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Create synthetic, unbalanced dataset
def create_data(num_samples_per_class):
    X = []
    y = []
    num_classes = 3
    for label in range(num_classes):
       for _ in range(num_samples_per_class[label]):
           img = np.random.rand(28, 28, 3) # Simple 28x28x3 image
           X.append(img)
           y.append(label)
    return np.array(X), np.array(y)

def augment_data(X, y, augment_class_index, multiplier):
  X_augmented = []
  y_augmented = []
  for i, label in enumerate(y):
    X_augmented.append(X[i])
    y_augmented.append(label)
    if label == augment_class_index:
      for _ in range(multiplier):
         augmented_img = X[i] + np.random.normal(0,0.05,X[i].shape) #simple gaussian blur
         X_augmented.append(augmented_img)
         y_augmented.append(label)
  return np.array(X_augmented), np.array(y_augmented)

num_samples_per_class = [500, 400, 50] # Class imbalance
X, y = create_data(num_samples_per_class)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Augment only class 2 by adding additional noisy images
X_train_augmented, y_train_augmented = augment_data(X_train, y_train, augment_class_index = 2, multiplier = 5)

# One-hot encode labels
y_train_augmented = tf.keras.utils.to_categorical(y_train_augmented, num_classes=3)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes=3)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax') # Output layer with 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_augmented, y_train_augmented, epochs=10, verbose=0)

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
actual_labels    = np.argmax(y_test, axis=1)

print(f"Predicted labels: {predicted_labels[0:50]}") #Improved predictions on the third class
print(f"Actual labels: {actual_labels[0:50]}")
```

In this scenario, the data augmentation step will increase the number of class 2 training samples, enabling the model to better learn features associated with class 2. This method is effective when the original training set contains a limited number of examples for a class. After training on the augmented data, the prediction probabilities will reflect a broader range of predictions, including the previously neglected class 2.

In closing, the "missing classes" in Keras models are rarely absent in terms of model architecture but rather absent from the model’s learned feature representation. These can be addressed effectively through techniques such as class weight balancing and data augmentation. Further exploration into techniques such as oversampling and undersampling may also be useful in some scenarios. For more detail on the specific implementations within TensorFlow, refer to the official TensorFlow documentation. Exploring the use of other loss functions, such as focal loss, which can also down-weight the well-classified samples and help learn underrepresented classes, may be useful. Finally, always remember to perform diligent error analysis of prediction outputs and consider model input and target distributions to improve overall performance.
