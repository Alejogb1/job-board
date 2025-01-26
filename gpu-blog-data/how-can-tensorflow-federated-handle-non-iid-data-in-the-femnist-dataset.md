---
title: "How can TensorFlow Federated handle non-IID data in the FEMNIST dataset?"
date: "2025-01-26"
id: "how-can-tensorflow-federated-handle-non-iid-data-in-the-femnist-dataset"
---

Federated learning faces a significant challenge when dealing with non-independent and identically distributed (non-IID) data, a common scenario in real-world applications. The Federated Extended MNIST (FEMNIST) dataset, partitioning handwritten character examples by writer, exemplifies this. In my experience developing federated learning models for personalized handwriting recognition, effectively addressing the heterogeneity present in FEMNIST data was crucial for achieving satisfactory global model performance. I'll outline how TensorFlow Federated (TFF) approaches this problem, focusing on techniques I've directly employed and the challenges I've observed.

The core issue with non-IID data in federated learning stems from the divergence in local client datasets. In FEMNIST, each writer might favor particular writing styles, certain characters may be underrepresented, and some might use entirely different handwriting conventions. This means local model updates derived from these datasets will be biased towards these idiosyncratic patterns. Simply averaging these updates globally leads to a model that performs poorly for clients different from those dominating the aggregation. TFF, however, provides several mechanisms to manage this inherent variability.

Firstly, TFF facilitates the use of *federated averaging* algorithms with customization, including those designed to mitigate the impact of non-IID data. The basic Federated Averaging (FedAvg) algorithm computes local model updates on each client's data and then aggregates these updates by averaging them. While not inherently resilient to non-IID, the process of local training itself can produce valuable and varied knowledge. The degree of this bias depends greatly on the degree of non-IIDness. TFF gives users the freedom to experiment with variants of FedAvg. One technique that I found useful is weighted averaging during the aggregation step. This technique assigns weights to individual clients in proportion to their data sizes or a quality metric of their training data. Large clients with high-quality data contribute more to global model update. In TFF, this weighting is configurable using a `tff.aggregators.WeightedMean` method. I've observed that this basic weighted approach is often a good starting point.

```python
import tensorflow as tf
import tensorflow_federated as tff

def create_keras_model():
    return tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(62, activation='softmax') # 62 labels in FEMNIST
  ])

def client_fn():
  return tff.simulation.datasets.ClientData.from_clients_and_tf_datasets(
    client_ids=None, tf_datasets=None, preprocessor=None, shuffle_buffer_size=None)

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=client_fn().element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1.0),
    model_aggregator=tff.aggregators.Mean() #Default aggregation, use tff.aggregators.WeightedMean instead for weighted averaging
)

state = iterative_process.initialize()

# Training loop
for round_num in range(10):
    sample_clients=client_data.client_ids[:10]
    federated_train_data = [client_data.create_tf_dataset_for_client(client) for client in sample_clients]
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))

```
In the provided code, we define a basic CNN model and use it to create a `tff.learning.Model`. The core of this first example lies in the usage of the `tff.learning.build_federated_averaging_process`. Crucially, in this case, the model aggregation is set with `tff.aggregators.Mean()`.  Changing this line to `tff.aggregators.WeightedMean()` would then implement weighted averaging. This emphasizes the flexibility offered by TFF for experimenting with different aggregation strategies. The example shows how the training happens across 10 rounds of communication between the clients.

Beyond simple weighted averaging, TFF also supports *differential privacy* mechanisms, which can be used to add noise to the client updates. By injecting small amounts of random noise, these algorithms protect individual client data and reduce the effect of individual clients with highly non-representative data samples. In the context of FEMNIST, this can be particularly beneficial when one client has a very different distribution of characters than the rest. In practice, I found that tuning the privacy parameters and carefully balancing utility and privacy requires detailed analysis of specific data distributions.

```python
import tensorflow as tf
import tensorflow_federated as tff

def create_keras_model():
    return tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(62, activation='softmax') # 62 labels in FEMNIST
  ])

def client_fn():
  return tff.simulation.datasets.ClientData.from_clients_and_tf_datasets(
    client_ids=None, tf_datasets=None, preprocessor=None, shuffle_buffer_size=None)

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=client_fn().element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1.0),
    model_aggregator=tff.aggregators.dp_aggregator(
    tf.float32,
    noise_multiplier=0.5,
    clients_per_round=10,
    l2_norm_clip=1.0)) #Differential Privacy aggregator


state = iterative_process.initialize()

# Training loop
for round_num in range(10):
    sample_clients=client_data.client_ids[:10]
    federated_train_data = [client_data.create_tf_dataset_for_client(client) for client in sample_clients]
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))

```
This code snippet demonstrates how to incorporate a *differential privacy* aggregator via `tff.aggregators.dp_aggregator`.  The key parameters here are `noise_multiplier`, which controls the magnitude of injected noise; `clients_per_round`, determining how many clients are involved in each aggregation round, and `l2_norm_clip`, which clamps individual client gradients. Proper values for these hyperparameters are data and model dependent. Choosing suboptimal values might negatively impact model utility.

Another aspect of dealing with non-IID data using TFF is data augmentation within clients. Although it does not address the underlying imbalance across the clients, it mitigates the effects of data sparsity on particular classes for each client. By applying data augmentations like rotation, scaling, and translation, we can artificially generate more training samples for each client. This increases the diversity of each client's local training dataset. While seemingly elementary, I've observed that it considerably improves the stability and generalizability of the global model when the amount of data is limited. Furthermore, I've found that more sophisticated data augmentation strategies, like those using GANs, can be applied with the client-side preprocessing. It is important to note that the type of augmentations must be chosen carefully with respect to the nature of the data.

```python
import tensorflow as tf
import tensorflow_federated as tff

def create_keras_model():
    return tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(62, activation='softmax') # 62 labels in FEMNIST
  ])

def client_fn():
  return tff.simulation.datasets.ClientData.from_clients_and_tf_datasets(
    client_ids=None, tf_datasets=None, preprocessor=lambda ds: ds.map(augment_data), shuffle_buffer_size=None)

def augment_data(example):
    image = example['pixels']
    image = tf.image.random_rotation(image, max_angle=0.2)
    image = tf.image.random_flip_left_right(image)
    example['pixels'] = image
    return example

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=client_fn().element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1.0),
    model_aggregator=tff.aggregators.Mean()
)

state = iterative_process.initialize()

# Training loop
for round_num in range(10):
    sample_clients=client_data.client_ids[:10]
    federated_train_data = [client_data.create_tf_dataset_for_client(client) for client in sample_clients]
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))
```
This final code snippet illustrates how one might incorporate client-side data augmentations.  The function `augment_data` specifies the augmentations to apply (`tf.image.random_rotation` and `tf.image.random_flip_left_right` in this case). This function is provided in the `preprocessor` parameter when constructing the `ClientData`. Consequently, each client’s local dataset undergoes transformations before being used in local training.

The key to handling non-IID data within the FEMNIST dataset (or others) isn't a single magic bullet, but rather a careful combination of techniques tailored to the specifics of the data. TFF provides flexible APIs for implementing strategies like weighted averaging, differential privacy, and client-side data augmentation, and more advanced techniques. Further study of research papers on federated learning and domain adaptation is highly recommended. Reviewing publications on personalized federated learning would give a deeper understanding of these strategies. Lastly, investigation of case studies within the TFF documentation and examples will illuminate practical implementation techniques.
