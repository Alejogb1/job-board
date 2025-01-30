---
title: "How to plot training and validation curves in Keras?"
date: "2025-01-30"
id: "how-to-plot-training-and-validation-curves-in"
---
Plotting training and validation curves during Keras model training provides crucial insights into model performance and potential issues like overfitting or underfitting.  My experience optimizing complex convolutional neural networks for medical image analysis heavily relied on meticulous monitoring of these curves.  A consistent observation across numerous projects highlighted the necessity of visualizing not just accuracy and loss, but also other relevant metrics for a complete understanding.


**1. Clear Explanation:**

Keras, inherently, doesn't directly offer a plotting function for training and validation curves.  However, its `fit` method returns a `History` object, which contains a dictionary of training metrics evaluated at the end of each epoch.  This dictionary serves as the foundation for generating the desired plots.  Effective visualization requires leveraging a plotting library like Matplotlib or Seaborn.  Typically, we extract the training and validation metrics from the `History` object and then utilize the plotting library's functions to create line plots, clearly distinguishing training and validation data.  The choice of metrics to plot depends on the specific problem, but commonly included are loss (e.g., categorical cross-entropy, mean squared error) and accuracy (or other relevant performance metrics such as precision, recall, F1-score, AUC, etc. depending on the problem).

Furthermore, careful attention must be given to the scale and labeling of the axes.  Clear legends are essential to distinguish training and validation data.  Finally, the resulting plot should provide a clear visual representation of how the model performs on both the training and validation data across epochs, allowing for easy identification of overfitting (validation performance plateaus or degrades while training performance continues to improve) or underfitting (both training and validation performance remain low).



**2. Code Examples with Commentary:**

**Example 1: Basic Accuracy and Loss Curves using Matplotlib**

```python
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'history' is the History object returned by model.fit()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Extract training and validation metrics
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the loss curves
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy curves
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

This example demonstrates the fundamental approach.  It directly accesses the 'loss' and 'accuracy' entries from the `history.history` dictionary.  The use of 'bo' (blue circles) and 'b' (blue line) helps distinguish training and validation data visually.  Error handling for missing metrics (which might occur if they weren't monitored during training) should be incorporated in a production setting.


**Example 2:  Handling Multiple Metrics with Seaborn**

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'history' is the History object returned by model.fit()
history_dict = history.history

# Convert history dictionary to DataFrame for easier handling
hist_df = pd.DataFrame(history_dict)

# Melt the DataFrame to long format for Seaborn
hist_df_melted = pd.melt(hist_df, id_vars=['epoch'], var_name='metric', value_name='value')


# Plot using Seaborn
sns.lineplot(x='epoch', y='value', hue='metric', data=hist_df_melted)
plt.title('Training and Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()
```

This utilizes Seaborn's capabilities to handle multiple metrics efficiently.  Converting the `history.history` dictionary to a Pandas DataFrame and then using the `melt` function allows for concise plotting of various metrics simultaneously.  Seaborn automatically handles legend creation, simplifying the process for scenarios involving numerous metrics like precision, recall, F1-score, etc.


**Example 3:  Customizable Plot with Enhanced Aesthetics**

```python
import matplotlib.pyplot as plt

# Assuming 'history' is the History object returned by model.fit()

metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']  #Define metrics to plot

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Training and Validation Metrics', fontsize=16)

for i, metric in enumerate(metrics):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    if 'loss' in metric:
        ax.plot(history.history[metric], label=metric)
        ax.plot(history.history['val_' + metric.split('_')[1]], label = 'val_' + metric.split('_')[1])
        ax.set_ylabel('Loss')
    else:
        ax.plot(history.history[metric], label=metric)
        ax.plot(history.history['val_' + metric.split('_')[1]], label = 'val_' + metric.split('_')[1])
        ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_title(metric.replace('_', ' ').title())
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

This example offers more granular control over the plot's appearance, utilizing subplots for better organization and employing custom labels and titles for enhanced clarity. It explicitly handles the plotting of both loss and accuracy metrics separately, improving readability, especially for reports and presentations. The `tight_layout` function prevents overlapping elements, maintaining a professional appearance.



**3. Resource Recommendations:**

The Matplotlib and Seaborn documentation are invaluable resources.  Explore the official documentation for both libraries, focusing on aspects such as customizing plots, handling multiple subplots, and leveraging different plot types.  Furthermore, consult general data visualization resources which emphasize best practices and effective communication through visuals.  Consider exploring books focusing on data visualization and Python plotting libraries for a deeper understanding of the subject.  Studying examples of effective visualizations in scientific publications can also significantly improve your skill.
