---
title: "How can I track epoch-wise losses in TensorFlow using MLflow?"
date: "2025-01-30"
id: "how-can-i-track-epoch-wise-losses-in-tensorflow"
---
The ability to diligently track epoch-wise losses during TensorFlow model training is crucial for effective model development and debugging. Leveraging MLflow for this purpose provides a structured and reproducible approach to experimentation. I've personally found this integration invaluable during several complex deep learning projects where pinpointing performance bottlenecks required examining loss trends across epochs. Specifically, a nuanced understanding of how to log these metrics with MLflow within a TensorFlow training loop helps reveal crucial details about learning dynamics.

The core mechanism involves instrumenting the TensorFlow training loop to calculate and log the loss at the end of each epoch. This requires accessing the loss values during training, which typically involves working with TensorFlow's built-in functionalities for loss calculation or those provided through custom loss functions. The key is that these losses must be extracted and then logged using MLflow's tracking API within the same loop. Furthermore, it’s critical to correctly use MLflow’s `log_metric` function, understanding its data types, and consistently linking the logged metrics to an active MLflow run for successful analysis.

Here's a concrete example demonstrating how to integrate MLflow loss tracking into a basic TensorFlow training loop. Suppose we have a standard training scenario where we iterate over a dataset, calculating the loss, and subsequently updating the model's weights.

**Example 1: Basic Epoch Loss Tracking**

```python
import tensorflow as tf
import mlflow

def train_model(model, train_dataset, optimizer, loss_fn, epochs=10):
    with mlflow.start_run():
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for images, labels in train_dataset:
                with tf.GradientTape() as tape:
                  predictions = model(images)
                  loss = loss_fn(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                epoch_loss += loss.numpy()
                num_batches +=1
            epoch_loss /= num_batches
            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)

            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

if __name__ == '__main__':
    # Example Usage
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    y_train = y_train[:].astype('float32')
    y_train = (y_train % 2).reshape(-1, 1)


    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

    train_model(model, train_dataset, optimizer, loss_fn)
```

In this example, the `train_model` function encompasses the training loop. The key operations occur within the loop: first, the loss is computed and aggregated for the epoch. Next, `mlflow.log_metric` records the average `epoch_loss`. Notice that `step=epoch` is explicitly provided to ensure that loss metrics are correctly associated with their respective epochs. This function logs the `epoch_loss` metric at each epoch, allowing MLflow to track its behavior across training. The averaging operation `epoch_loss /= num_batches` is important to make the total loss independent of the mini-batch size. Finally, `mlflow.start_run()` ensures that the training session is properly recorded as a separate run within MLflow, which allows us to organize our experiments.

The use of `tf.GradientTape()` within the training loop is standard practice for dynamic computation of gradients. The specific training process of forward-propagation, loss calculation, back-propagation, and weight update is abstracted from MLflow-related functionalities. This allows the loss tracking logic to be integrated into any type of model without significant changes to the training flow.

Consider scenarios where, instead of training for a fixed number of epochs, I need a training regimen with early stopping based on validation loss. This requires a slightly different approach, where the validation loss must be tracked alongside the training loss and then used in conjunction with a callback to terminate training if a certain condition is met.

**Example 2: Tracking Both Training and Validation Losses**

```python
import tensorflow as tf
import mlflow
import numpy as np
def train_model_early_stopping(model, train_dataset, val_dataset, optimizer, loss_fn, patience=3, epochs=10):
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    with mlflow.start_run():
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            num_train_batches = 0
            for images, labels in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = model(images)
                    loss = loss_fn(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                epoch_train_loss += loss.numpy()
                num_train_batches += 1

            epoch_train_loss /= num_train_batches
            mlflow.log_metric("epoch_train_loss", epoch_train_loss, step=epoch)


            epoch_val_loss = 0.0
            num_val_batches = 0
            for images, labels in val_dataset:
                predictions = model(images)
                loss = loss_fn(labels, predictions)
                epoch_val_loss += loss.numpy()
                num_val_batches+=1

            epoch_val_loss /= num_val_batches
            mlflow.log_metric("epoch_val_loss", epoch_val_loss, step=epoch)

            print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")


            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


if __name__ == '__main__':
    # Example Usage (reuse the previous model)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    y_train = y_train[:].astype('float32')
    y_train = (y_train % 2).reshape(-1, 1)

    x_val = x_val.reshape(10000, 784).astype('float32') / 255.0
    y_val = y_val[:].astype('float32')
    y_val = (y_val % 2).reshape(-1, 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    train_model_early_stopping(model, train_dataset, val_dataset, optimizer, loss_fn)
```
In the second example, I have extended the previous training loop to include validation loss calculation within each epoch, using a separate `val_dataset`. Importantly, I log both the `epoch_train_loss` and the `epoch_val_loss`, again associating each with its corresponding epoch through the `step` argument. Early stopping is implemented by tracking the validation loss and terminating training if the validation loss does not improve for a specified number of epochs (`patience`). This demonstrates how MLflow can be used to not only track metrics but also to inform the termination of training.

For cases where model architecture or other training parameters need to be tracked along with the losses, MLflow parameters tracking becomes essential. Let's examine an example where a hyperparameter, specifically the learning rate of the optimizer, is tracked in addition to the epoch loss.

**Example 3: Tracking Learning Rate as a Parameter**
```python
import tensorflow as tf
import mlflow
import numpy as np
def train_model_with_lr(model, train_dataset, optimizer, loss_fn, learning_rate, epochs=10):
    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate)
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for images, labels in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = model(images)
                    loss = loss_fn(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_loss += loss.numpy()
                num_batches += 1

            epoch_loss /= num_batches
            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

if __name__ == '__main__':
    # Example Usage
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    learning_rate = 0.005
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    y_train = y_train[:].astype('float32')
    y_train = (y_train % 2).reshape(-1, 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    train_model_with_lr(model, train_dataset, optimizer, loss_fn, learning_rate)
```

In the third example, the learning rate is explicitly tracked using `mlflow.log_param`. By recording these hyperparameters, I’m able to associate a specific hyperparameter configuration with the loss trends I observe over epochs when I examine the runs through MLflow's UI, which is very helpful when conducting experiments. This allows more rigorous examination of which parameter settings have resulted in the most stable learning curves and convergence for a certain training session.

For effective utilization of MLflow with TensorFlow, I would recommend a deeper study of the MLflow API documentation, specifically focusing on tracking functionalities, the use of metrics and parameters, and how to organize nested runs effectively. Additionally, I find a thorough review of best practices for data logging within TensorFlow helpful when training complex models. Finally, familiarity with experiment tracking platforms is generally useful to leverage the tools effectively, and resources exploring different MLflow functionalities for complex experiment tracking would be valuable.
