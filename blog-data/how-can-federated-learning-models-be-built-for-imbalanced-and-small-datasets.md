---
title: "How can federated learning models be built for imbalanced and small datasets?"
date: "2024-12-23"
id: "how-can-federated-learning-models-be-built-for-imbalanced-and-small-datasets"
---

Let's tackle this problem; it's a challenge I've seen crop up in more projects than I'd prefer. Building federated learning (FL) models, particularly when dealing with imbalanced and small datasets, is definitely a hurdle. I recall a particularly tricky project involving sensor data from a dispersed network of devices. We faced exactly this problem: each device had a limited amount of data, and some devices had significantly more data than others for specific events we were trying to model. It became clear that standard FL approaches wouldn't cut it.

The core problem is that federated learning assumes a degree of data homogeneity, and sufficient data points across clients. When that falls apart, two major issues arise: *model bias* and *slow convergence*. With imbalanced datasets, models tend to gravitate towards learning the distribution of the dominant class on each device, potentially disregarding valuable information from underrepresented classes. The small dataset size further exacerbates this, as there isn't enough local data for each device to train a robust model independently. Centralized solutions wouldn’t work due to privacy concerns.

So, how do we tackle this? It isn't as simple as tweaking a hyperparameter. Several techniques, often used in combination, can be employed. First and foremost, *data augmentation* plays a critical role. While the raw data size at each device might be small, careful augmentation techniques can increase the effective training data size. However, we must ensure we’re not introducing bias through augmentation. For time-series sensor data, for instance, methods such as jittering, scaling, time warping, or permutation could add diversity without changing the underlying signal. This isn't about adding noise; it's about adding variability that is realistic for the dataset.

The second technique focuses on *weighted aggregation*. In a standard federated averaging algorithm, local model updates are weighted equally based on the number of data points on each device. This favors the clients with larger datasets, which, with imbalanced data, can be detrimental. I've found that using a weighted average based on the *class balance* on each device, rather than just dataset size, is incredibly effective. If a device has far fewer samples of one class, its updates associated with that class should carry more weight in the global model aggregation. This requires each device to communicate some minimal class information along with model updates.

Thirdly, a shift in learning strategy can be helpful. Instead of solely relying on local learning with global aggregation via federated averaging, we can incorporate techniques like *transfer learning* and *meta-learning*. Transfer learning involves leveraging a pre-trained model on a related task or dataset. This mitigates the effects of a small dataset size since the model already has some understanding of the underlying data distribution. Meta-learning, on the other hand, aims to learn how to adapt quickly to new tasks or datasets. This approach can be particularly useful when there is high heterogeneity in the data distributions across the devices. It learns parameters that are more generalizable from a meta-training phase.

Let's illustrate this with some practical examples. Consider a classification problem where our data is a set of images on different devices. Here's how data augmentation could be applied in Python, using `tensorflow`:

```python
import tensorflow as tf

def augment_image(image):
  """Applies basic data augmentation to an image."""
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_brightness(image, max_delta=0.1)
  image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
  return image

# Example of how it can be incorporated within a data pipeline:
def create_dataset(images, labels, batch_size, augmentation=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if augmentation:
      dataset = dataset.map(lambda image, label: (augment_image(image), label),
                            num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

This `augment_image` function adds some basic transformations, and it’s included in the dataset preparation. The augmentation details depend on the specific data type, but the idea remains the same: increasing the diversity and robustness of the local datasets.

Now, let's look at the concept of class-weighted federated averaging, using `numpy` for simplicity of demonstration. In a real-world scenario, these would involve actual model updates, but the core idea is the same:

```python
import numpy as np

def weighted_average(client_updates, client_sizes, class_balances):
    """Calculates the weighted average of client updates based on class balances."""
    global_update = np.zeros_like(client_updates[0])
    total_weight = 0

    for i, update in enumerate(client_updates):
        # Assuming class_balances[i] is a dictionary of class: count
        class_weights = np.array(list(class_balances[i].values())) / sum(class_balances[i].values())
        # Invert weights for imbalanced classes (minor classes = more weight).
        class_weights = 1 / (class_weights + 1e-7) # Add small value to avoid division by zero
        weighted_update = update * np.sum(class_weights) # Class weight sum
        global_update += weighted_update
        total_weight += np.sum(class_weights)

    return global_update / total_weight
# Example Usage:
client_updates = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
client_sizes = [100, 50, 20]
class_balances = [{0: 10, 1: 90}, {0: 30, 1: 20}, {0: 5, 1: 15}] # example imbalances
global_model_update = weighted_average(client_updates, client_sizes, class_balances)

print(f"Global model update: {global_model_update}")
```

In the `weighted_average` function, we use inverse class frequencies as weights for each device’s model updates; a device with fewer instances of a class contributes more to the global update related to that class. This is a simplified approach, but it highlights the core idea. In real code, this function would work with PyTorch tensors or TensorFlow variables, depending on the chosen framework.

Finally, to illustrate a simplified transfer learning approach in `keras` (for easier demonstrability), here's an example, though a more practical setup would include more intricate layer freezing or fine-tuning:

```python
from tensorflow import keras
from tensorflow.keras import layers

def build_transfer_learning_model(input_shape, num_classes, base_model):
  """Builds a model by adding a classification layer to a pretrained model."""

  base_model.trainable = False  # Freeze base model layers

  inputs = keras.Input(shape=input_shape)
  x = base_model(inputs, training = False)
  x = layers.GlobalAveragePooling2D()(x)
  outputs = layers.Dense(num_classes, activation="softmax")(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

# example usage:
input_shape = (224, 224, 3) # Example image input
num_classes = 10

base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape,
)

transfer_model = build_transfer_learning_model(input_shape, num_classes, base_model)
```

This code uses a pre-trained MobileNetV2 model as a feature extractor and adds a classification layer on top. Here, we freeze the pre-trained model's weights to focus the training on the final classification layer, which is suited to our specific problem. This allows for quicker training when dealing with limited datasets.

These are, naturally, highly simplified illustrations. In practice, implementing these methods requires careful consideration of hyperparameter tuning, the specific characteristics of the data, and the chosen federated learning framework. Also, things like differential privacy and secure multi-party computation must often be considered if the data has privacy concerns.

For further study, I highly recommend *Federated Learning* by Qiang Yang, et al. This book covers the fundamental aspects of federated learning and presents a comprehensive overview of different algorithms and applications. Also, exploring recent papers on *Meta-Learning for Imbalanced Data* could be insightful. In addition, the *Deep Learning* book by Goodfellow, Bengio, and Courville contains valuable theory and practices for the techniques I mentioned. These resources offer both foundational knowledge and advanced techniques related to this problem. Addressing the unique challenges posed by federated learning with imbalanced and small datasets requires a careful blend of techniques. It is not a straightforward matter and requires an iterative approach with detailed evaluation. I’ve learned over time that experimentation is key.
