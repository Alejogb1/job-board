---
title: "How do I compute multi-class accuracy correctly in Keras?"
date: "2024-12-23"
id: "how-do-i-compute-multi-class-accuracy-correctly-in-keras"
---

Let's tackle multi-class accuracy in Keras, something I've grappled with quite a bit over the years. A common pitfall, I've found, is not fully understanding how Keras handles categorical data and metrics under the hood. Misinterpreting the output, particularly when dealing with one-hot encoded labels versus sparse categorical ones, can lead to inaccurate assessments. I recall a specific project where we were classifying images into a dozen categories, and the initial reported accuracy was, shall we say, surprisingly high – and alarmingly misleading. It prompted a deep dive into Keras’ metrics and how they interacted with our data format. So, what’s the proper approach?

The critical first step revolves around understanding your output data format. Keras provides two primary ways of expressing multi-class labels: one-hot encoded vectors and sparse integer labels. *One-hot encoded vectors* are what you typically get when using `to_categorical` from `keras.utils`. Here, each label is represented as a vector of zeros, except for a single '1' at the index corresponding to the class. *Sparse integer labels*, on the other hand, are simply integers representing each class index. For example, class 2 would be represented as the integer `2`.

The `accuracy` metric in Keras, or even the more specific `categorical_accuracy`, expects one-hot encoded labels when evaluating predictions from a model that has a softmax activation in its output layer. The key is ensuring that the predicted probabilities from the softmax layer are correctly compared against the one-hot encoded representation of the true labels. For this case, it's pretty straightforward, as Keras handles the argmax calculation internally.

Here's a snippet to demonstrate the correct way of utilizing `categorical_accuracy` when our true labels are one-hot encoded:

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

# Dummy Data
num_classes = 5
input_shape = (10,)
batch_size = 32

X_train = np.random.rand(1000, *input_shape)
y_train = np.random.randint(0, num_classes, size=1000)
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)

X_test = np.random.rand(200, *input_shape)
y_test = np.random.randint(0, num_classes, size=200)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)


# Model creation
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=input_shape),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=[metrics.CategoricalAccuracy()])

model.fit(X_train, y_train, batch_size=batch_size, epochs=5, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print(f"Test Accuracy (categorical_accuracy): {accuracy:.4f}")
```

Notice that our `y_train` and `y_test` are converted to one-hot encodings via `keras.utils.to_categorical`, and we use `metrics.CategoricalAccuracy()` during compilation. If you were to use sparse integer labels with a softmax output, this wouldn't work correctly.

Now, let's delve into the scenario where you have sparse integer labels instead of one-hot vectors. This often happens when you're dealing directly with class indices or when loading datasets already in this format. Using `categorical_accuracy` in this scenario will provide inaccurate readings because it expects a one-hot encoded format. Here's where `sparse_categorical_accuracy` becomes crucial. This metric takes the argmax of the predicted probabilities and compares it against the sparse representation of the true labels directly, skipping the one-hot decoding step. It correctly infers that it should not be expecting probabilities for one-hot encoded vector rather single value.

Here’s how to modify the prior example to utilize `sparse_categorical_accuracy` with sparse integer labels:

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

# Dummy Data
num_classes = 5
input_shape = (10,)
batch_size = 32

X_train = np.random.rand(1000, *input_shape)
y_train = np.random.randint(0, num_classes, size=1000)  # No to_categorical here
X_test = np.random.rand(200, *input_shape)
y_test = np.random.randint(0, num_classes, size=200)    # No to_categorical here

# Model creation
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=input_shape),
    layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer=optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=[metrics.SparseCategoricalAccuracy()])

model.fit(X_train, y_train, batch_size=batch_size, epochs=5, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print(f"Test Accuracy (sparse_categorical_accuracy): {accuracy:.4f}")
```

Key changes: the `to_categorical` step is removed and `metrics.SparseCategoricalAccuracy` is used, as well as `sparse_categorical_crossentropy` for the loss function. It's crucial to also change the loss function here; if you kept `categorical_crossentropy` while providing sparse labels, the results would again be invalid. Choosing the appropriate loss function depends heavily on your label format.

Finally, another area where I've seen confusion arises is when trying to calculate accuracy manually, especially if you are trying to diagnose why a model is underperforming. For example, you might find you're getting different values using a custom calculation than what Keras reports, which leads to debugging headaches. If we intend to calculate it manually, we must take care to properly interpret the output.

Here’s an example of manually calculating accuracy and how to match that with `sparse_categorical_accuracy`:

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

# Dummy Data
num_classes = 5
input_shape = (10,)
batch_size = 32

X_train = np.random.rand(1000, *input_shape)
y_train = np.random.randint(0, num_classes, size=1000)
X_test = np.random.rand(200, *input_shape)
y_test = np.random.randint(0, num_classes, size=200)


# Model creation
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=input_shape),
    layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer=optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=[metrics.SparseCategoricalAccuracy()])


model.fit(X_train, y_train, batch_size=batch_size, epochs=5, verbose=0)
y_pred_probabilities = model.predict(X_test, batch_size=batch_size, verbose=0)

y_pred = np.argmax(y_pred_probabilities, axis=-1)
manual_accuracy = np.mean(y_pred == y_test)

loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)


print(f"Test Accuracy (sparse_categorical_accuracy): {accuracy:.4f}")
print(f"Manual Accuracy: {manual_accuracy:.4f}")

```

In this snippet, after obtaining the probability output from the model, we use `np.argmax(y_pred_probabilities, axis=-1)` to convert the probabilities into the predicted classes. After that, we compare our predicted classes with our true labels in the test set and find the average using `np.mean` to get our manual accuracy. Comparing this with the built-in `sparse_categorical_accuracy` should yield almost the exact result.

For a deeper understanding, I’d highly recommend exploring chapter 6, "Deep Learning for Text," from *Deep Learning with Python* by François Chollet. This chapter covers multi-class classification and categorical encoding in quite a bit of detail. Also, the official Keras documentation provides excellent explanations of loss functions and metrics, and going through those explanations carefully is quite beneficial. I also suggest reviewing the original papers on softmax and cross-entropy loss – those often provide much-needed theoretical background that can help clarify how they interact with one another.

To summarize, ensure you:
1.  Understand your label format: Are they one-hot encoded or sparse integer labels?
2.  Use the correct Keras metric: `categorical_accuracy` for one-hot encoded labels and `sparse_categorical_accuracy` for sparse integer labels.
3.  Choose the correct loss function that fits the format of your labels
4. Be aware how you are computing metrics manually.

By paying attention to these details, you’ll compute multi-class accuracy correctly and avoid the kind of headaches I’ve encountered over my years of machine learning work. Always double check your data format and ensure that you’re consistently interpreting the outputs. It can save you from having to backtrack and rework your entire model later down the line.
