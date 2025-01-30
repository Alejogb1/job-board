---
title: "How does TensorFlow support federated learning?"
date: "2025-01-30"
id: "how-does-tensorflow-support-federated-learning"
---
Federated learning, where a shared model is trained across decentralized devices holding sensitive data, presents a unique challenge to traditional machine learning paradigms. TensorFlow addresses this through the `TensorFlow Federated` (TFF) library, offering a specialized API designed for simulating and deploying federated computations. My experience with TFF, predominantly in developing personalized medical diagnosis models, highlights its efficacy in managing heterogeneous data sources while preserving user privacy.

TFF operates under a principle of separating the *federated computation* from the underlying *implementation*. This abstraction is critical. The core of a federated learning process can be described as follows:
1. **Client Selection:** A subset of available clients, each possessing unique data, is chosen for a given round of training.
2. **Local Training:** Each selected client trains the global model on its local data, generating updates to the model's weights.
3. **Aggregation:** The updates from each client are aggregated (often through averaging or weighted averaging) to produce a new, improved global model.
4. **Model Update:** The improved global model is redistributed to clients for the next round of local training. This cycle repeats until convergence.

TFF provides the necessary tools to construct and execute these federated computations with minimal exposure to the complexities of communication and distributed training inherent in such a process. It uses a functional programming paradigm, where computations are expressed as pure functions without side effects, facilitating optimization and maintainability.

The primary building blocks within TFF are the `tff.tf_computation` and `tff.federated_computation` decorators. `tff.tf_computation` transforms standard TensorFlow code into a callable function that executes as a part of a federated computation. This allows us to leverage familiar TensorFlow operations within the federated setting. Conversely, `tff.federated_computation` describes the actual federated process across client devices.

To illustrate, let us consider training a simple linear regression model in a federated setting.

```python
import tensorflow as tf
import tensorflow_federated as tff

# 1. Define the model using TF
@tff.tf_computation
def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    return model

# 2. Define a loss function.
@tff.tf_computation
def loss_fn(model, data):
    x = data[0]
    y = data[1]
    y_pred = model(x)
    return tf.reduce_mean(tf.square(y_pred - y))

# 3. Define a client training function.
@tff.tf_computation
def client_update_fn(model, data, learning_rate=0.1):
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    with tf.GradientTape() as tape:
        loss = loss_fn(model, data)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model

# 4. Define a federated averaging function for server updates.
@tff.federated_computation(
    tff.type_at_server(tff.type_at_clients(tff.keras.models.Model)),
    tff.type_at_clients(tff.type_at_clients((tf.float32, tf.float32))),
)
def server_update_fn(server_model, client_data):
  client_models = tff.federated_map(client_update_fn, (server_model, client_data))
  averaged_model = tff.federated_mean(client_models)
  return averaged_model

# 5. Define the federated training loop function.
@tff.federated_computation
def training_process(client_data):
    server_model = tff.federated_broadcast(model_fn()) # Broadcast initial model
    new_server_model = server_update_fn(server_model, client_data)
    return new_server_model

# 6. Create dummy client data
client_data = [((tf.constant([[1.0],[2.0]]),tf.constant([[2.0],[4.0]])),),
                ((tf.constant([[3.0],[4.0]]),tf.constant([[6.0],[8.0]])),)]

# 7. Execute a round of federated training
trained_model = training_process(client_data)

#The type of the server model is returned from the training loop.
print(f"Trained Server Model Type: {trained_model.type_signature}")

```
In this first example, I define a `model_fn` and `loss_fn` using TensorFlow’s Keras API, then decorate them with `tff.tf_computation`. The `client_update_fn` calculates gradients and updates model weights locally. The `server_update_fn` performs federated averaging of client models. Finally, `training_process` initializes the model and executes a single federated update round. Dummy client data is manually created and fed to the training loop. Note that the `tff.federated_map` allows to apply `client_update_fn` to each client independently. `tff.federated_mean` handles aggregation of the resulting updated models. The type signatures of inputs and outputs to federated computations must be explicit. This ensures correct computation behavior across devices.

The next code example shows data loading with the Federated EMNIST dataset, which serves as a representative federated dataset.

```python
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

# Load the Federated EMNIST dataset.
emnist_train, _ = tff.simulation.datasets.emnist.load_data()

# Select a sample of clients (e.g., the first 2 clients)
sample_clients = emnist_train.client_ids[:2]
sample_dataset = [emnist_train.create_tf_dataset_for_client(client) for client in sample_clients]

# Function to preprocess the data
def preprocess(ds):
    def _reshape(el):
        return tf.reshape(el['pixels'], (-1, 784)), el['label']
    return ds.batch(20).map(_reshape)

# Apply the preprocessing step
preprocessed_sample = [preprocess(ds) for ds in sample_dataset]

# Example of accessing preprocessed data
first_dataset = next(iter(preprocessed_sample[0]))
print(f"Shape of first batch of data from first client: {first_dataset[0].shape}")

# Convert to appropriate format (using list of tuples)
converted_sample = [((x_batch, y_batch),) for ds in preprocessed_sample for x_batch, y_batch in ds]

print(f"Shape of converted data format: {len(converted_sample)} with batch shape {converted_sample[0][0][0].shape}")

# Run the previous federated algorithm using the converted data
trained_model = training_process(converted_sample)
print(f"Trained Model Type: {trained_model.type_signature}")
```

Here, the EMNIST data is loaded and a preprocessing function is applied to batch and reshape the data for a dense layer as in the previous example. This example highlights TFF's support for standard federated datasets and how to transform this data into the expected format, which in this case is a nested structure of `((feature, label),)`. This structure aligns with how we defined the input types for `server_update_fn` in example 1.

Finally, let us explore how to use a custom Keras model within TFF.
```python
import tensorflow as tf
import tensorflow_federated as tff

#Define the Model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(784,))
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# Define the model using TF
@tff.tf_computation
def custom_model_fn():
    model = MyModel()
    return model

# Define a loss function.
@tff.tf_computation
def custom_loss_fn(model, data):
    x = data[0]
    y = data[1]
    y_pred = model(x)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, y_pred))

# Define a client training function.
@tff.tf_computation
def custom_client_update_fn(model, data, learning_rate=0.1):
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    with tf.GradientTape() as tape:
        loss = custom_loss_fn(model, data)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model


# Define a federated averaging function for server updates.
@tff.federated_computation(
    tff.type_at_server(tff.type_at_clients(tff.keras.models.Model)),
    tff.type_at_clients(tff.type_at_clients((tf.float32, tf.int32))),
)
def custom_server_update_fn(server_model, client_data):
  client_models = tff.federated_map(custom_client_update_fn, (server_model, client_data))
  averaged_model = tff.federated_mean(client_models)
  return averaged_model


# Define the federated training loop function.
@tff.federated_computation
def custom_training_process(client_data):
    server_model = tff.federated_broadcast(custom_model_fn()) # Broadcast initial model
    new_server_model = custom_server_update_fn(server_model, client_data)
    return new_server_model


# Run the federated training loop with the sample_dataset generated from previous example
trained_model = custom_training_process(converted_sample)
print(f"Trained Model Type: {trained_model.type_signature}")
```
This final example uses a custom Keras model subclass `MyModel` within the TFF framework. The `custom_model_fn`, `custom_loss_fn`, and `custom_client_update_fn` illustrate that even custom model architectures can be seamlessly integrated using the `@tff.tf_computation` decorator. Notice that `sparse_categorical_crossentropy` was used for the loss and the label tensors in client data are now of `tf.int32` type, to align with the loss.

For additional resources, I recommend exploring the following official channels:
*   TensorFlow Federated API Documentation.
*   TensorFlow official tutorials and guides.
*   Examples of specific federated algorithms within the TFF repository.

In summary, TensorFlow's federated learning support is provided through the `TensorFlow Federated` library, offering an abstraction that facilitates the creation and management of federated computations while leveraging the power and flexibility of TensorFlow's core capabilities. Using `tff.tf_computation` and `tff.federated_computation`, one can implement federated learning algorithms, process federated datasets, and use a wide variety of different Keras architectures and training procedures. The key lies in understanding the functional programming paradigm and the data types expected by the federated computations.
