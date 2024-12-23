---
title: "How can TensorFlow model checkpoints be used to plot matplotlib graphs?"
date: "2024-12-23"
id: "how-can-tensorflow-model-checkpoints-be-used-to-plot-matplotlib-graphs"
---

Alright,  You're asking how to leverage TensorFlow model checkpoints to create matplotlib visualizations, and that's a really practical question that I've certainly encountered in my time. It's not as simple as a direct one-to-one mapping, but understanding the interplay between TensorFlow's saving mechanisms and matplotlib's plotting capabilities opens up some powerful debugging and analysis techniques. Think of it as taking snapshots of your model's internal state, then translating those snapshots into visual stories.

My experience with this dates back to a project involving recurrent neural networks for sequence modeling. I found that simply observing the loss values wasn't giving me the full picture. I needed a deeper look at how the model’s learned representations were evolving, especially across training epochs. That's where the checkpoint-to-matplotlib pipeline proved invaluable. The key here isn't about directly using the checkpoint files to drive matplotlib; rather, it's about extracting the model weights, biases, or other relevant tensors from the checkpoint and then using *those* values to construct plots.

First off, let’s clarify what a TensorFlow checkpoint actually is. It's not some magical database you can query; it's essentially a collection of binary files that store the serialized tensors of your model’s trainable variables. Think of it as a photograph of your model at a particular point in training. TensorFlow provides tools to save and load these checkpoints, and we are going to use those to our benefit.

Now, onto the 'how'. We’re fundamentally dealing with a two-step process. Step one involves loading the data from the checkpoint, which requires a TensorFlow model instance and the ability to restore variables from it. Step two involves using matplotlib to create our visualizations based on these restored values.

Let's jump into some concrete code snippets to illustrate this process.

**Example 1: Plotting Layer Weights as Histograms**

Let's assume you have a simple fully connected layer model. A useful visualization would be to see the distribution of the weights at different epochs. Here's how we might approach that:

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def plot_weights_histogram(checkpoint_dir, layer_name, saved_epoch=0, restore_epoch=0):
    """Plots a histogram of weights for a given layer using model checkpoints.

    Args:
        checkpoint_dir: Path to the directory containing the checkpoints.
        layer_name: The name of the layer whose weights we want to plot.
        saved_epoch: The epoch when the model was trained.
        restore_epoch: The specific epoch we are restoring (default=0).
    """

    # build model instance
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,), name="input_layer"),
        tf.keras.layers.Dense(10, activation='softmax', name="output_layer")
    ])

    # Create the checkpoint manager
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # Find all checkpoint files
    if restore_epoch == 0:
        latest_checkpoint = manager.latest_checkpoint
    else:
       latest_checkpoint = os.path.join(checkpoint_dir, f"ckpt-{restore_epoch}")

    # Check if there is a checkpoint to restore from
    if latest_checkpoint:
        ckpt.restore(latest_checkpoint)
        print(f'Restored from {latest_checkpoint}')
    else:
        print("No checkpoint found for this epoch. Cannot restore.")
        return

    # Get layer weights
    for layer in model.layers:
        if layer.name == layer_name:
            weights = layer.weights[0]  # Assuming first element holds the weights

            # Plot the histogram
            plt.figure(figsize=(10, 6))
            plt.hist(weights.numpy().flatten(), bins=50, color='skyblue', edgecolor='black')
            plt.title(f'Weights Histogram - Layer: {layer_name} - Epoch: {saved_epoch}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(checkpoint_dir, f"weights_histogram_epoch_{saved_epoch}.png"))
            plt.show()
            return
    print("layer not found")


# Example usage:
if __name__ == '__main__':
   # Assuming you have a checkpoint dir at ./checkpoints
   checkpoint_dir = './checkpoints'
   layer_to_plot = 'input_layer'  # The name of the layer to plot

   # Call the function to plot the weights of the input layer at the restore epoch.
   plot_weights_histogram(checkpoint_dir, layer_to_plot, saved_epoch=20, restore_epoch=20)
```

This example demonstrates how to load a model from a checkpoint, locate a specific layer, extract its weight tensor, and then visualize it as a histogram using matplotlib. Notice that `tf.train.CheckpointManager` is used to locate the correct checkpoint file. The `saved_epoch` refers to the epoch that the graph corresponds to (i.e. 20), while `restore_epoch` is set to the specific epoch we want to plot (also 20 here) to avoid issues with mismatched epochs.

**Example 2: Plotting Learning Rate over Time**

Many optimizers use a learning rate schedule that changes across training. Monitoring this learning rate can be essential. Here's how we might visualize that, again using checkpoints:

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def plot_learning_rate(checkpoint_dir, saved_epoch=0, restore_epoch=0):
    """Plots the learning rate evolution during training, given checkpoint directory.

    Args:
        checkpoint_dir: Path to the directory containing the checkpoints.
        saved_epoch: The epoch when the model was trained.
        restore_epoch: The specific epoch we are restoring (default=0).
    """

    # Build model and optimizer instance
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,), name="input_layer"),
        tf.keras.layers.Dense(10, activation='softmax', name="output_layer")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Set initial learning rate

    # Create the checkpoint manager
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # Find all checkpoint files
    if restore_epoch == 0:
        latest_checkpoint = manager.latest_checkpoint
    else:
        latest_checkpoint = os.path.join(checkpoint_dir, f"ckpt-{restore_epoch}")


    # Check if there is a checkpoint to restore from
    if latest_checkpoint:
        ckpt.restore(latest_checkpoint)
        print(f'Restored from {latest_checkpoint}')
    else:
        print("No checkpoint found for this epoch. Cannot restore.")
        return

    # Get Learning Rate from optimizer
    learning_rate = optimizer.learning_rate.numpy()

    # Plot learning rate (as a single point)
    plt.figure(figsize=(8, 5))
    plt.scatter(saved_epoch, learning_rate, color='purple', s=100)
    plt.title(f'Learning Rate at Epoch {saved_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, f"learning_rate_at_epoch_{saved_epoch}.png"))
    plt.show()

# Example usage:
if __name__ == '__main__':
   # Assuming you have a checkpoint dir at ./checkpoints
   checkpoint_dir = './checkpoints'

   # Call the function to plot the learning rate of epoch 20
   plot_learning_rate(checkpoint_dir, saved_epoch=20, restore_epoch=20)
```

This code extracts and plots the learning rate from the optimizer stored within the checkpoint. Again, `tf.train.CheckpointManager` handles finding the correct checkpoint, and it is critical that the optimizer object used in training is also being checkpointed.

**Example 3: Visualizing Activation Maps**

For convolutional neural networks (CNNs), visualizing activation maps can help diagnose what features the network is paying attention to. This is a slightly more involved task, but still achievable using checkpoints:

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_activation_map(checkpoint_dir, layer_name, image_index=0, saved_epoch=0, restore_epoch=0):
    """Plots activation map for a layer using model checkpoints.

    Args:
        checkpoint_dir: Path to the directory containing the checkpoints.
        layer_name: The name of the convolutional layer whose activation we want to plot.
        image_index: The index of the image in the dataset for which to generate the map.
        saved_epoch: The epoch when the model was trained.
        restore_epoch: The specific epoch we are restoring (default=0).
    """

    # Load a sample image (for simplicity)
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    sample_image = x_train[image_index].astype(np.float32) / 255.0
    sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension

    # Build model
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv_layer_1'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_layer_2'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation='softmax')
     ])

    # Create the checkpoint manager
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # Find all checkpoint files
    if restore_epoch == 0:
        latest_checkpoint = manager.latest_checkpoint
    else:
       latest_checkpoint = os.path.join(checkpoint_dir, f"ckpt-{restore_epoch}")

    # Check if there is a checkpoint to restore from
    if latest_checkpoint:
        ckpt.restore(latest_checkpoint)
        print(f'Restored from {latest_checkpoint}')
    else:
        print("No checkpoint found for this epoch. Cannot restore.")
        return

    # Create a model to get intermediate layer activation
    activation_model = tf.keras.models.Model(inputs=model.input,
                                             outputs=model.get_layer(layer_name).output)

    # Get activations
    activations = activation_model.predict(np.expand_dims(sample_image, axis=-1)) # Add channel dimension

    # Plot activation map (choose a channel to display, e.g., the first one)
    plt.figure(figsize=(8, 8))
    plt.imshow(activations[0, :, :, 0], cmap='viridis')
    plt.title(f'Activation Map - Layer: {layer_name}, Epoch {saved_epoch}')
    plt.axis('off')
    plt.savefig(os.path.join(checkpoint_dir, f"activation_map_epoch_{saved_epoch}.png"))
    plt.show()


# Example usage:
if __name__ == '__main__':
   # Assuming you have a checkpoint dir at ./checkpoints
   checkpoint_dir = './checkpoints'
   layer_to_plot = 'conv_layer_2'

   # Call the function to plot the activation map of the conv_layer_2 at the restore epoch.
   plot_activation_map(checkpoint_dir, layer_to_plot, image_index=5, saved_epoch=20, restore_epoch=20)
```

This demonstrates a method for visualizing activations at specific layers by creating a secondary model specifically for feature map extraction. Keep in mind that activation maps can be high dimensional and can require techniques for visualization of many channels at once.

**Important Considerations and Further Resources**

-   **Checkpoint Structure:** Be aware that different checkpoint formats exist in TensorFlow (e.g., v1, v2). The examples I provided are based on the `tf.train.Checkpoint` format common in modern TensorFlow.
-   **TensorFlow Version:** The exact functions and their behavior might vary slightly across TensorFlow versions. Always consult the official TensorFlow documentation.
-   **Data Visualization Principles:** While matplotlib is a powerful tool, remember the basics of effective data visualization. Choose appropriate plot types, label axes correctly, and ensure your visualizations are informative.

For deeper learning, I'd highly recommend the following resources:

*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron**: Provides comprehensive practical guidance on TensorFlow and machine learning in general. Check the chapters regarding Model Persistence and Visualization.
*   **The Official TensorFlow Documentation:** This is your go-to resource for any specific API details. Look for the sections on saving and loading models using the checkpoint api and using eager execution.
*   **Research papers on network visualization techniques:** Search for papers on "activation maximization", "saliency maps," and related topics. These go beyond basic plotting and provide ideas for deeper network analysis.
*   **The Keras documentation:** If you are working with a Keras Model instance, consult the documentation for Keras related API calls.

In essence, plotting graphs from checkpoints involves a structured approach: defining your model architecture, restoring the desired variables, and using matplotlib to express the extracted data visually. Don't think of it as a direct connection between checkpoints and plots but rather as a process of extracting model state and then visualizing those extracted pieces. It takes some practice, but it's a very useful skill to have in your machine learning toolkit.
