---
title: "How can I plot loss and accuracy after training on my datasets?"
date: "2025-01-30"
id: "how-can-i-plot-loss-and-accuracy-after"
---
The efficacy of a machine learning model is fundamentally judged by its performance on unseen data, which is directly reflected in its loss and accuracy metrics during and after training.  Successfully visualizing these metrics is crucial for understanding model convergence, identifying overfitting or underfitting, and ultimately optimizing the training process.  In my experience, consistent monitoring and plotting of loss and accuracy curves have been invaluable for debugging and improving model performance across various projects, ranging from image classification to natural language processing.  The key to effective plotting lies in choosing the appropriate libraries and structuring the data in a way that facilitates clear and informative visualizations.

My approach typically involves utilizing the training history object returned by many popular machine learning frameworks, such as TensorFlow/Keras and PyTorch. These history objects store metrics collected during each epoch of the training process.  We then leverage data visualization libraries like Matplotlib to create insightful plots.  Below, I will illustrate this process with examples focusing on Keras, a high-level API for TensorFlow.

**1. Clear Explanation:**

The training process iteratively adjusts model parameters to minimize a loss function, which quantifies the difference between the model's predictions and the actual target values.  Accuracy, on the other hand, measures the percentage of correctly classified samples.  Plotting these metrics against the number of training epochs provides a visual representation of the model's learning curve.  An ideal learning curve shows a decreasing loss and increasing accuracy as the number of epochs increases, eventually plateauing as the model converges. Deviations from this ideal curve – for instance, a consistently high loss or fluctuating accuracy – often indicate problems like insufficient training data, inappropriate model architecture, or hyperparameter misconfigurations.

To achieve this visualization, we first need to access the training history. Keras' `model.fit()` method returns a `History` object, which contains a dictionary storing the loss and accuracy metrics for each epoch.  We then extract this data and utilize Matplotlib's plotting functionalities to generate the desired graphs.  Furthermore, the graphs should include clear labels for axes and a descriptive title to ensure readability and facilitate interpretation.  Appropriate choice of scale on the axes is crucial for avoiding misleading visualization.  For example, a logarithmic scale might be preferred for the loss if it spans several orders of magnitude.

**2. Code Examples with Commentary:**

**Example 1: Basic Loss and Accuracy Plot**

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Assume 'X' and 'y' are your feature and target data respectively.
# This is a simplification; replace with your actual data loading.
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

This example demonstrates a basic plot of training and validation loss and accuracy. The use of `validation_data` during model fitting is crucial for monitoring generalization performance and detecting overfitting.  The legend clarifies which line represents training and validation data.  The `figsize` argument controls the size of the plot for better readability.


**Example 2:  Handling Multiple Metrics**

```python
import matplotlib.pyplot as plt
# ... (Previous code, data loading and model definition remain the same) ...

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall']) # Adding more metrics

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Training and Validation Precision')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Training and Validation Recall')
plt.legend()

plt.tight_layout()
plt.show()
```

This example extends the previous one to plot precision and recall in addition to loss and accuracy.  This allows for a more comprehensive evaluation of the model's performance, particularly in imbalanced datasets. The use of subplots effectively organizes multiple metrics within a single figure.


**Example 3:  Using Logarithmic Scale for Loss**

```python
import matplotlib.pyplot as plt
import numpy as np
# ... (Previous code, data loading and model definition remain the same) ...

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.semilogy(history.history['loss'], label='Training Loss') #Using semilogy for y-axis
plt.semilogy(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Log Scale)')
plt.legend()

# ... (Accuracy plot remains the same) ...

plt.tight_layout()
plt.show()
```

This example demonstrates the use of a logarithmic scale (`semilogy`) for the loss plot, which is beneficial when the loss values span several orders of magnitude. This improves the visualization of the loss curve, particularly in the early stages of training.



**3. Resource Recommendations:**

For deeper understanding of Keras and its functionalities, I highly recommend consulting the official Keras documentation. Matplotlib's documentation provides extensive tutorials and examples for creating various types of plots. A solid understanding of linear algebra and calculus is essential for grasping the underlying principles of loss functions and optimization algorithms.  Finally, exploration of diverse machine learning datasets and models from reputable sources will enhance practical experience.
