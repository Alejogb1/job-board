---
title: "Why does MobileNet exhibit low accuracy in federated learning on the MNIST dataset compared to a TensorFlow environment, and how can this be improved?"
date: "2025-01-30"
id: "why-does-mobilenet-exhibit-low-accuracy-in-federated"
---
The discrepancy in MobileNet accuracy observed between a centralized TensorFlow training environment and a federated learning (FL) setting on MNIST arises primarily from the limitations inherent in FL's decentralized data access and the architectural nuances of MobileNet, specifically its dependence on batch statistics and sensitivity to data heterogeneity. I've personally encountered this issue while attempting to deploy a MobileNet-based image classification model across several simulated edge devices, each with access to a non-representative subset of the MNIST training data. The stark performance difference initially prompted a deep dive into the interplay between model architecture and the constraints imposed by federated learning.

The core challenge lies in the fact that federated learning training operates on local data subsets held by clients, often without centralized access. This introduces two significant complications: *statistical heterogeneity* (non-IID data) and *limited local data quantity*. MNIST, despite being a seemingly simple dataset, can be partitioned among clients in a non-random, skewed manner, resulting in highly biased local datasets. MobileNet, a convolutional neural network (CNN) designed for efficient execution on mobile devices, relies heavily on batch normalization (BN). BN layers compute and normalize activations using the mean and standard deviation calculated within each batch. In a typical centralized setting, a well-mixed training dataset ensures that each batch, on average, provides a relatively stable representation of the global data distribution. However, in federated learning, this is not guaranteed. The batch statistics computed on local, possibly skewed, client data significantly differ from those obtained from the aggregate, global distribution. Further, local batch statistics within each client can be highly unstable given the small size of each client’s local dataset, resulting in inaccurate normalization. This instability of batch normalization statistics propagates throughout the MobileNet architecture, causing gradient calculation errors, destabilizing the training process, and contributing to the observed lower accuracy.

Furthermore, the architecture of MobileNet, while efficient for mobile deployment, was not inherently designed for handling the complexities of federated settings. Specifically, the reliance of many of the depthwise convolutions on batch statistics adds additional instability when the client local data is not well representative of the global data distribution. This issue is exacerbated when client data has a severe class imbalance or is biased towards specific examples. Unlike larger, more parameter-rich models, MobileNet's smaller size means that even a minor destabilization of the batch normalization layer can more significantly impact the model's overall performance. Therefore, the combination of MobileNet’s reliance on batch statistics and the federated learning environment's data access patterns results in significant accuracy degradation.

To mitigate these issues, several strategies can be employed. A primary focus must be on addressing the instability introduced by batch normalization. One approach is to replace Batch Normalization (BN) layers with Group Normalization (GN) or Layer Normalization (LN). GN divides the channels into groups and normalizes within each group, while LN normalizes across the features of a single sample. These techniques are not as dependent on batch statistics, offering improved stability in federated settings where batch sizes may be limited and data distributions are heterogeneous. Another essential strategy is to apply client-side regularization. While regularization is common in all deep learning, in FL, it helps prevent clients from overfitting on their small data sets, improving generalization across the whole federated system. Regularization can be applied via weight decay or dropout layers. Also, a more sophisticated training regime can be employed. The most common algorithm is FedAvg. Alternative algorithms, such as FedProx, modify the loss function to reduce the effect of client heterogeneity by introducing a proximal term.

Here are a few code examples to illustrate how to improve the situation in practice:

**Example 1: Replacing Batch Normalization with Group Normalization**

This example shows how to implement a MobileNet-like network in TensorFlow and switch batch normalization to group normalization within the convolution blocks. It focuses on model architecture modification to improve model performance in the context of data heterogeneity.

```python
import tensorflow as tf

def conv_block(inputs, filters, kernel_size, strides, use_groupnorm=False, groups=32):
  x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
  if use_groupnorm:
    x = tf.keras.layers.GroupNormalization(groups=groups)(x)
  else:
      x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.ReLU()(x)
  return x


def depthwise_separable_conv(inputs, filters, strides, use_groupnorm=False, groups=32):
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(inputs)
    if use_groupnorm:
       x = tf.keras.layers.GroupNormalization(groups=groups)(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
    if use_groupnorm:
       x = tf.keras.layers.GroupNormalization(groups=groups)(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def create_mobilenet_like(input_shape=(28, 28, 1), num_classes=10, use_groupnorm=False, groups=32):
  inputs = tf.keras.Input(shape=input_shape)
  x = conv_block(inputs, filters=32, kernel_size=(3, 3), strides=2, use_groupnorm=use_groupnorm, groups=groups)
  x = depthwise_separable_conv(x, filters=64, strides=1, use_groupnorm=use_groupnorm, groups=groups)
  x = depthwise_separable_conv(x, filters=128, strides=2, use_groupnorm=use_groupnorm, groups=groups)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

model_with_gn = create_mobilenet_like(use_groupnorm=True, groups=8)
model_with_bn = create_mobilenet_like(use_groupnorm=False)
```

In this example, the `create_mobilenet_like` function demonstrates how to construct a simplified version of MobileNet with configurable batch normalization. This can be toggled to use either group normalization or batch normalization. By calling the function with `use_groupnorm=True`, all the normalization layers within the model use group normalization instead of batch normalization. This substitution will result in better stability and performance within a federated learning environment.

**Example 2: Client-Side Regularization with Weight Decay**

This example demonstrates how to implement weight decay in client-side training for the MobileNet model. The regularization process helps to prevent local overfitting and improve generalization in the federated setting.

```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def create_client_optimizer(learning_rate, weight_decay):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    return optimizer

def client_train(model, client_dataset, num_epochs, optimizer):
    for _ in range(num_epochs):
        for batch_x, batch_y in client_dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_x, training=True)
                loss = tf.keras.losses.CategoricalCrossentropy()(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Create a sample dataset for demonstration purposes
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
x_train = x_train[..., tf.newaxis]

client_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)


# Assume you have `model`, created via `create_mobilenet_like`
model = create_mobilenet_like(use_groupnorm=True, groups=8) # Initialize with Group Normalization

# Configure a weight decay value and learning rate
learning_rate = 0.001
weight_decay = 0.0005
optimizer = create_client_optimizer(learning_rate, weight_decay)

# Perform training on the client data using the custom training function.
client_train(model, client_data, num_epochs=3, optimizer=optimizer)

```

This code illustrates the client-side training process within a federated setting using custom training loops. The `create_client_optimizer` function uses `AdamW`, which natively supports weight decay, enhancing the model's generalization to unseen data. During the local training phase, weight decay is applied using `AdamW`, a variant of the Adam optimizer that incorporates weight decay, to add to the loss function. The `client_train` function loops over the `client_data` and applies the local optimizer with the regularizing weight decay on the model’s weights, promoting better convergence.

**Example 3: Implementing a Proximal Term in the Loss Function (FedProx)**

This code snippet demonstrates how to apply a proximal term in the loss function, which will help mitigate the negative impacts of client data heterogeneity and contribute to model convergence.

```python
import tensorflow as tf

def fedprox_client_train(model, global_model_weights, client_dataset, num_epochs, optimizer, mu=0.01):
    for _ in range(num_epochs):
        for batch_x, batch_y in client_dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_x, training=True)
                loss = tf.keras.losses.CategoricalCrossentropy()(batch_y, predictions)

                # Proximal term for FedProx
                proximal_term = 0.0
                for local_var, global_var in zip(model.trainable_variables, global_model_weights):
                    proximal_term += tf.reduce_sum(tf.square(local_var - global_var))

                proximal_term *= (mu / 2.0)  # Apply the proximal weight
                total_loss = loss + proximal_term

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Same setup as previous example
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
x_train = x_train[..., tf.newaxis]

client_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)


# Assuming you have `model`, and `global_model`
model = create_mobilenet_like(use_groupnorm=True, groups=8)
global_model = create_mobilenet_like(use_groupnorm=True, groups=8)
# Initialize with Group Normalization. Get global weights from a model previously trained
global_model.set_weights(model.get_weights()) # Mock global weights

learning_rate = 0.001
weight_decay = 0.0005
optimizer = create_client_optimizer(learning_rate, weight_decay)

# Perform training with FedProx
fedprox_client_train(model, global_model.trainable_variables, client_data, num_epochs=3, optimizer=optimizer, mu=0.1)
```

This code snippet shows the core logic behind the FedProx algorithm. The main alteration is the introduction of the `proximal_term` which is computed using the global and local model weights. During each gradient update the `total_loss` (the sum of the cross entropy loss and proximal term) is used to update the model weights. The `mu` parameter balances the impact of the proximal term. This regularization helps to keep local updates close to the global model and can reduce the performance decrease associated with heterogeneous data distributions.

To further enhance model accuracy in this federated setting, I recommend exploring resources on federated optimization techniques. Specifically, consider research papers and books focusing on algorithms like FedProx, SCAFFOLD, and variations of adaptive learning rates in federated learning. Also, investigate data augmentation and more advanced model regularization techniques. An understanding of how these strategies can be implemented with MobileNet and other CNNs will significantly improve performance in FL scenarios.
