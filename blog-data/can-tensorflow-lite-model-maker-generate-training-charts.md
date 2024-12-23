---
title: "Can TensorFlow Lite Model Maker generate training charts?"
date: "2024-12-23"
id: "can-tensorflow-lite-model-maker-generate-training-charts"
---

Okay, let’s tackle this. I remember a project a few years back, working on a mobile app for plant recognition. We were using TensorFlow Lite for on-device inference, and getting the model training just *so* was crucial for performance. One of the hurdles we faced was understanding the training dynamics without constantly printing to the console; the classic “blind spot” scenario. While TensorFlow Lite Model Maker is excellent at streamlining the model creation and conversion process, the direct generation of training charts, as you might expect from, say, a dedicated training dashboard in TensorFlow proper, isn’t a feature it directly offers in the way people commonly think. It's not a graphical UI spitting out curves. However, it doesn't mean all is lost, we can definitely get visual insight into what's happening during training using other methods, and it's worth knowing the landscape.

Let's unpack that. Model Maker is designed to simplify the process of creating and converting TensorFlow models specifically for deployment on resource-constrained devices. It focuses on the *outcome*: a quantized and optimized *tflite* file ready for use. It abstracts away many of the complexities of the training pipeline using high-level APIs. This means we, the users, don’t have a fine-grained control over every training detail. However, the training metrics are still accessible, but we need to extract them programmatically. The Model Maker API does provide access to metrics from within the training loop that can then be leveraged to generate the visualizations you are seeking. This requires a bit of intermediate coding, but it's all very manageable. The trick is capturing these values during the training process, and then using a plotting library like `matplotlib` or `seaborn` to visualize them.

Here’s a practical example using a common image classification scenario. Let’s assume you are using Model Maker to train an image classifier. The training API typically yields history information after calling `model.fit()`. Here’s a code snippet:

```python
import tensorflow as tf
import tensorflow_lite_model_maker as tflm
import matplotlib.pyplot as plt

# Assume your data is loaded and preprocessed as 'train_data'
# And the model is defined as 'model' using tflm.image_classifier.create
# For example:
# train_data = tflm.DataLoader.from_folder(data_path)
# model = tflm.image_classifier.create(train_data, ...)

# Here, let's pretend the model is already created
def train_and_plot(model, train_data):
    history = model.fit(train_data, epochs=10) # Train for 10 epochs

    # Extract training metrics
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    val_loss = history.history.get('val_loss')  # Check if validation data is used
    val_accuracy = history.history.get('val_accuracy')

    epochs = range(1, len(loss) + 1)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    if val_loss:  # Plot if there is validation data
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'r', label='Training Accuracy')
    if val_accuracy:
        plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    return model

trained_model = train_and_plot(model, train_data)

```

This code captures the loss and accuracy during training, and then uses `matplotlib` to generate a combined graph showing loss and accuracy curves for both training and validation, if you’ve provided validation data during training. The `history` object returned by the `fit` function is key here. It’s a dictionary that keeps track of the different metrics as the training progresses, we are essentially just pulling that data and displaying it via a visualization.

The first example uses a basic `matplotlib` plotting functionality for illustration, but there are other more refined options available. We can incorporate `seaborn` for example to improve the style and feel of the plot. Here’s a modified snippet to showcase that, with seaborn integrated:

```python
import tensorflow as tf
import tensorflow_lite_model_maker as tflm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming data loading and model creation steps are the same as above
# model, train_data are available.
def train_and_plot_seaborn(model, train_data):
  history = model.fit(train_data, epochs=10)

  # Convert history into a pandas DataFrame for easier seaborn handling
  metrics = history.history
  metrics['epoch'] = range(1, len(metrics['loss']) + 1) # adding epoch for the x axis
  metrics_df = pd.DataFrame(metrics)
  
  # Melt the DataFrame for easier plotting with seaborn
  metrics_df_melted = metrics_df.melt(id_vars=['epoch'], var_name='metric', value_name='value')

  # Create plot using seaborn
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=metrics_df_melted, x='epoch', y='value', hue='metric', style='metric')
  plt.title('Training Metrics')
  plt.xlabel('Epochs')
  plt.ylabel('Value')
  plt.show()

  return model

trained_model = train_and_plot_seaborn(model, train_data)

```

This version uses `pandas` to restructure the training metrics into a "long" format, then leverage's `seaborn` to perform the line plot. The result is often a more visually appealing and informative graph, particularly when multiple metrics are being displayed. The use of `seaborn` offers a lot of control over the look and feel of the visualizations, so there's room to customize these charts to match your specific needs.

Finally, if you're working with extremely large datasets, you might want to periodically save the training metrics rather than waiting until the end of the `fit` call. That would enable tracking and plotting in real-time or quasi-real-time during lengthy training processes. The callback interface in Keras which underlies Model Maker is useful here and can be leveraged within model maker. Below is an example callback function that will save the metrics at the end of each epoch:

```python
import tensorflow as tf
import tensorflow_lite_model_maker as tflm
import matplotlib.pyplot as plt
import numpy as np


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MetricsCallback, self).__init__()
        self.epoch_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_metrics.append(logs)

    def get_metrics_data(self):
      return self.epoch_metrics

def train_and_plot_callback(model, train_data):
    metrics_callback = MetricsCallback()
    model.fit(train_data, epochs=10, callbacks=[metrics_callback])
    metrics_data = metrics_callback.get_metrics_data()
    
    metrics = {'epoch': [],'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}
    
    for index, item in enumerate(metrics_data):
      metrics['epoch'].append(index + 1)
      metrics['loss'].append(item['loss'])
      metrics['accuracy'].append(item['accuracy'])

      if 'val_loss' in item:
         metrics['val_loss'].append(item['val_loss'])
      else:
         metrics['val_loss'].append(np.nan)

      if 'val_accuracy' in item:
         metrics['val_accuracy'].append(item['val_accuracy'])
      else:
         metrics['val_accuracy'].append(np.nan)

    #Plotting using matplotlib similar to the first example
    epochs = metrics['epoch']
    loss = metrics['loss']
    accuracy = metrics['accuracy']
    val_loss = metrics['val_loss']
    val_accuracy = metrics['val_accuracy']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    return model


trained_model = train_and_plot_callback(model, train_data)
```
This version implements a custom callback class `MetricsCallback` that captures training metrics at the end of each epoch, stores the captured metrics, and has a get method to return all the metrics which can then be used to plot the graphs as in the previous example. This approach is beneficial for longer training processes or when you need the metrics at different stages of the training process, not just the final result. The ability to tap into the Keras callback functionality is incredibly powerful when used with model maker since it allows for intermediate processing of the metrics as desired.

To further your understanding of these topics, I would highly recommend these resources. First, for a comprehensive grasp of the underlying TensorFlow, the *TensorFlow 2.0 Tutorial* by Martin Görner is an excellent resource. It explains the fundamentals and the practical applications of the platform. Second, delving into the Keras documentation, specifically the sections on model training and callbacks, can greatly help understanding how `fit` works and how to extract training metrics as demonstrated above. For data visualization techniques, I recommend *Python Data Science Handbook* by Jake VanderPlas. It covers `matplotlib` and `seaborn` in detail and provides a solid foundation for creating meaningful data visualizations. Also, the `pandas` library documentation is great for better understanding how to manage data for plotting.

In summary, while TensorFlow Lite Model Maker doesn’t directly generate charts itself, you absolutely *can* achieve the same effect with a bit of code. Extracting training history, using `matplotlib`, `seaborn`, or other libraries and potentially even using callbacks, offers the flexibility and insight you need to understand your model training dynamics. It's about understanding the output of the API and then building the tooling needed to get the visual understanding you seek.
