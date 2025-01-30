---
title: "Should plot be overwritten each epoch?"
date: "2025-01-30"
id: "should-plot-be-overwritten-each-epoch"
---
Directly overwriting plots during each training epoch of a machine learning model can severely hinder the debugging process and limit the depth of analysis obtainable from training visualizations. I've personally encountered significant difficulties trying to understand the dynamics of complex models, especially those with unstable training phases, when using this overwrite strategy. Instead of overwriting, creating a series of plots or appending data to a single plot, where each corresponds to a specific epoch, provides a far richer and more useful visual history.

A critical aspect of model training, often overlooked, is the analysis of trends and transitions in key metrics, such as loss and accuracy, as the model iterates through the dataset. Overwriting a plot at each epoch effectively collapses this temporal dimension, removing the ability to observe these crucial evolutionary patterns. Imagine a scenario where the validation loss plateaus after an initial decrease, a situation that frequently indicates a model is approaching its performance limit. If plots are overwritten, the history of this initial decrease, and how it might correlate with changes in training loss, is lost. Similarly, transient spikes or anomalies in metrics, which can reveal issues like poor learning rate selection or exploding gradients, become practically invisible when the visualization is constantly updated in place.

An alternative strategy, which I have adopted in my workflow, involves creating a new plot for each epoch or, preferably, appending the metric data points to an existing plot and then updating the figure, allowing me to track changes over training time. This approach not only preserves the history of training but also facilitates the comparison of performance across different epochs. Furthermore, visualizing the data sequentially can reveal subtleties in training dynamics such as oscillatory behavior or instances of overfitting that could otherwise be masked by the summary view of a single, overwritten plot.

Consider a typical training process that monitors loss and accuracy on both training and validation sets. An approach that overwrites the plots might look something like this, where matplotlib.pyplot is used.

```python
import matplotlib.pyplot as plt
import numpy as np

def train_model(epochs, data):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Simulate training calculations
        train_loss = np.random.rand() * (1 / (epoch + 1))
        val_loss = np.random.rand() * (1 / (epoch + 1) + 0.1)
        train_acc = 1 - train_loss
        val_acc = 1 - val_loss


        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Overwriting plot: This is the problematic part
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.title(f'Loss - Epoch {epoch}')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label='Training Accuracy')
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.title(f'Accuracy - Epoch {epoch}')
        plt.legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
```

The above example represents the scenario in which a plot is created, displayed, and then closed at each epoch. While it appears to function in a live sense and shows updates, the previous epoch's plot is completely lost. The code itself is also inefficient due to the constant creation of figure objects. The `plt.close()` is essential to free the resources, but also directly causes the issue. This methodology makes it impossible to track overall progress effectively.

To address these shortcomings, I've implemented approaches where data points are appended to an existing plot. One such approach utilizes matplotlib's interactive mode to update the plot after each epoch:

```python
import matplotlib.pyplot as plt
import numpy as np


def train_model_live(epochs, data):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    loss_lines = axes[0].plot([], [], label='Training Loss', color='blue')
    loss_lines.extend(axes[0].plot([], [], label='Validation Loss', color='red'))
    axes[0].set_title('Loss')
    axes[0].legend()

    acc_lines = axes[1].plot([], [], label='Training Accuracy', color='blue')
    acc_lines.extend(axes[1].plot([], [], label='Validation Accuracy', color='red'))
    axes[1].set_title('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show(block=False)


    for epoch in range(epochs):
        # Simulate training calculations
        train_loss = np.random.rand() * (1 / (epoch + 1))
        val_loss = np.random.rand() * (1 / (epoch + 1) + 0.1)
        train_acc = 1 - train_loss
        val_acc = 1 - val_loss

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        loss_lines[0].set_data(range(len(train_loss_history)), train_loss_history)
        loss_lines[1].set_data(range(len(val_loss_history)), val_loss_history)

        acc_lines[0].set_data(range(len(train_acc_history)), train_acc_history)
        acc_lines[1].set_data(range(len(val_acc_history)), val_acc_history)


        for ax in axes:
           ax.relim()
           ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.ioff()
    plt.show()
```

In this implementation, the `plt.ion()` command puts matplotlib in interactive mode and allows for live plot updates without having to recreate a window with every new data point. This method efficiently updates the plot data while retaining the complete training progression.  The data series are updated via `set_data` and the axes are automatically scaled using `relim` and `autoscale_view`. The `draw` and `flush_events` call ensure the screen updates. This approach avoids the computational overhead of creating a completely new figure for every epoch.  After the training loop is done, `plt.ioff` stops interactive mode, and the finalized plot is displayed.

An even more structured way of handling the plotting is to save a separate figure of the training process for each epoch, in addition to any interactive visualization you might have. This method provides an archival record of your training runs.

```python
import matplotlib.pyplot as plt
import numpy as np
import os


def train_model_save_plots(epochs, data, output_dir='training_plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []


    for epoch in range(epochs):
        # Simulate training calculations
        train_loss = np.random.rand() * (1 / (epoch + 1))
        val_loss = np.random.rand() * (1 / (epoch + 1) + 0.1)
        train_acc = 1 - train_loss
        val_acc = 1 - val_loss

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)


        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.title(f'Loss - Epoch {epoch}')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label='Training Accuracy')
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.title(f'Accuracy - Epoch {epoch}')
        plt.legend()

        plt.tight_layout()
        filename = os.path.join(output_dir, f'epoch_{epoch}.png')
        plt.savefig(filename)
        plt.close()
```

This approach iterates over each epoch and generates a separate PNG image, saved within a dedicated directory, providing a complete record. The plot generated at every epoch now has an associated physical file. This facilitates a comprehensive offline review of the training procedure later. I prefer to use this method in conjunction with a real-time interactive plot.

In conclusion, overwriting plots each epoch creates a myopic view of model training. Instead, appending plot data or saving separate plots for each epoch offers a more granular perspective of training dynamics. This approach enhances debugging capabilities, assists in identifying nuanced issues during model training, and promotes a deeper understanding of a model's learning process. I recommend exploring resources detailing matplotlib's interactive plotting capabilities and object-oriented architecture, as well as exploring guides on best practices for visualizing model training results.
