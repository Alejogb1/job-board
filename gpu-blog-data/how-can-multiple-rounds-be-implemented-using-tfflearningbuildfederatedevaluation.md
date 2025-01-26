---
title: "How can multiple rounds be implemented using tff.learning.build_federated_evaluation?"
date: "2025-01-26"
id: "how-can-multiple-rounds-be-implemented-using-tfflearningbuildfederatedevaluation"
---

Implementing multiple rounds of evaluation using `tff.learning.build_federated_evaluation` requires careful consideration of how to manage state across rounds and, crucially, how to properly feed data to the evaluation process. The core challenge is that `build_federated_evaluation` generates a *single* TensorFlow graph for a single round of evaluation, not a continuous process. We'll need to orchestrate the execution of this graph iteratively and aggregate results accordingly.

I've encountered this exact scenario frequently when fine-tuning models in a federated learning setting, where I needed periodic insight into model performance without impacting the training loop unnecessarily. The key is to understand that `build_federated_evaluation` provides a function that returns a callable, the `federated_eval` function, which takes the model and client data as arguments. It’s this callable that we must execute repeatedly.

Here's how I typically approach it, broken down into the process and then illustrated with code examples:

1. **Establish the Foundation:** We must first define a model function and client datasets in the way expected by `tff.learning.build_federated_evaluation`. Crucially, ensure the client datasets are structured such that we can iterate through them across multiple evaluation rounds.

2. **Initialize the Evaluation Function:** Use `tff.learning.build_federated_evaluation` to create the `federated_eval` function. This function encapsulates the logic for a single round of federated evaluation.

3. **Iterate Through Rounds:** Enclose the call to `federated_eval` within a loop. Each iteration of the loop constitutes a distinct round of evaluation. Within the loop, we need to extract the appropriate slice of client data for the current round, which might be achieved by moving an index through the client datasets.

4. **Aggregate Results:** Each execution of `federated_eval` will yield a dictionary of metrics. We must accumulate these metrics across rounds, typically by summing them or averaging them after the loop completes, depending on the specific nature of the metrics. For metrics like losses which need to be averaged across dataset sizes, we need to keep track of the number of samples evaluated in each round.

5. **Handle Model Updates:** Importantly, if the model is being trained, the evaluated model should be updated based on that training before being passed to the evaluation function in the next round. This often involves retrieving model weights after the training step and assigning them to the evaluated model before the next call to `federated_eval`.

Now, let's look at some code illustrating this process. Assume we've defined the necessary utility functions for a simple model and client data construction (not shown for brevity):

```python
import tensorflow as tf
import tensorflow_federated as tff

# Assume we have model_fn and make_client_data defined elsewhere.
# These generate a simple Keras model and some random client datasets for testing.
# For illustrative purposes, we'll use a simplified version that assumes
# fixed dataset sizes for each round across all clients.
def model_fn():
    # Define your Keras model here
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def make_client_data(num_clients, samples_per_client):
  # Simplified creation to have a fixed dataset size per client
  datasets = []
  for i in range(num_clients):
    features = tf.random.normal((samples_per_client, 10))
    labels = tf.random.uniform((samples_per_client, 1), maxval=2, dtype=tf.int32)
    datasets.append(tf.data.Dataset.from_tensor_slices((features, labels)).batch(32))
  return datasets

# Example usage
NUM_CLIENTS = 2
SAMPLES_PER_CLIENT = 100
ROUND_COUNT = 3
BATCH_SIZE = 32


# Create the federated evaluation function
evaluation = tff.learning.build_federated_evaluation(model_fn)

client_datasets = make_client_data(NUM_CLIENTS, SAMPLES_PER_CLIENT)


def evaluate_over_rounds(evaluation_function, client_data, num_rounds, batch_size):
    """Evaluates a model over multiple rounds.

    Args:
        evaluation_function: A function returned by
            tff.learning.build_federated_evaluation.
        client_data: A list of tf.data.Datasets, each representing a client.
        num_rounds: The number of evaluation rounds to run.
    Returns:
        A dictionary of aggregated metrics.
    """
    model = model_fn()  # Obtain the initial model for evaluation

    total_metrics = {}

    for round_num in range(num_rounds):
        # Create the current round datasets by slicing
        current_round_data = [ds for ds in client_data] # In this simplified version, use the full client datasets
        # Convert the list of datasets to a TFF dataset
        federated_dataset = tff.simulation.datasets.TestClientData(current_round_data)
        
        # Perform the evaluation for this round and update metrics
        round_metrics = evaluation_function(model, federated_dataset)
        
        # Process metrics for this round
        for metric_name, metric_value in round_metrics.items():
            if metric_name not in total_metrics:
                total_metrics[metric_name] = 0.0

            # Aggregate metrics.  Here, we simply sum them for the purpose of illustration,
            # for metrics like loss, you'd often need to average them (after
            # tracking the number of samples per round).
            total_metrics[metric_name] += metric_value

    # Average metrics based on the number of rounds
    for metric_name, value in total_metrics.items():
      total_metrics[metric_name] = value/num_rounds
    return total_metrics


# Perform the multi-round evaluation
aggregated_metrics = evaluate_over_rounds(evaluation, client_datasets, ROUND_COUNT, BATCH_SIZE)
print("Aggregated evaluation metrics:", aggregated_metrics)

```

In this example, the `evaluate_over_rounds` function encapsulates the core logic for multiple round evaluation. I create a new instance of the model before the evaluation begins. Within the evaluation loop, I then use the `federated_eval` function to perform evaluation on all the clients in each round. This simple illustration assumes that all clients provide their complete datasets to each evaluation round, which, while not always the case in real-world applications, keeps the focus on the multi-round logic. The example aggregates the results by summing each metric across rounds then averaging them. In a real scenario, metrics like losses would need to be accumulated differently based on the number of samples contributing to that loss. Also, I am using a basic client data creation for simplification to showcase the round implementation which will not typically be the case in real world applications.

Let’s add a second example that shows how to actually deal with multiple evaluation rounds by iterating over the dataset and a more meaningful metric aggregation:

```python
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def model_fn():
    # Define your Keras model here
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def make_client_data(num_clients, samples_per_client):
  # Simplified creation to have a fixed dataset size per client
  datasets = []
  for i in range(num_clients):
    features = tf.random.normal((samples_per_client, 10))
    labels = tf.random.uniform((samples_per_client, 1), maxval=2, dtype=tf.int32)
    datasets.append(tf.data.Dataset.from_tensor_slices((features, labels)).batch(32))
  return datasets


# Example usage
NUM_CLIENTS = 2
SAMPLES_PER_CLIENT = 100
ROUND_COUNT = 3
BATCH_SIZE = 32


# Create the federated evaluation function
evaluation = tff.learning.build_federated_evaluation(model_fn)

client_datasets = make_client_data(NUM_CLIENTS, SAMPLES_PER_CLIENT)


def evaluate_over_rounds_iterative(evaluation_function, client_data, num_rounds, batch_size):
  """Evaluates a model over multiple rounds using an iterator.

    Args:
        evaluation_function: A function returned by
            tff.learning.build_federated_evaluation.
        client_data: A list of tf.data.Datasets, each representing a client.
        num_rounds: The number of evaluation rounds to run.
    Returns:
        A dictionary of aggregated metrics.
  """

  model = model_fn()
  total_metrics = {}
  num_examples = 0

  # Wrap the client data into iterators
  client_iterators = [iter(dataset) for dataset in client_data]


  for round_num in range(num_rounds):
        # Retrieve the next chunk for each client
        current_round_data = []
        for it in client_iterators:
            try:
              current_round_data.append(next(it))
            except StopIteration:
              # We ran out of samples from the dataset for this client, restart the iterator
              print("Restarting iterator, this should not happen in this example")
              client_iterators = [iter(dataset) for dataset in client_data]
              current_round_data.append(next(it)) # Retrieve the first batch again

        federated_dataset = tff.simulation.datasets.TestClientData(current_round_data)
        
        # Perform the evaluation for this round
        round_metrics = evaluation_function(model, federated_dataset)

        # Aggregate metrics (summed, then averaged)
        for metric_name, metric_value in round_metrics.items():
          if metric_name not in total_metrics:
            total_metrics[metric_name] = 0.0

          total_metrics[metric_name] += metric_value * sum(np.prod(x.shape) for x in current_round_data[0][0]) # Weight the metric value by the number of samples in the datasets


        num_examples += sum(np.prod(x.shape) for x in current_round_data[0][0])
  
  # Average the metrics across all of the examples
  for metric_name, value in total_metrics.items():
      total_metrics[metric_name] = value/num_examples

  return total_metrics


# Perform the multi-round evaluation
aggregated_metrics = evaluate_over_rounds_iterative(evaluation, client_datasets, ROUND_COUNT, BATCH_SIZE)
print("Aggregated evaluation metrics:", aggregated_metrics)

```

In this second example, `evaluate_over_rounds_iterative`, each round uses only the next batch of client data, retrieved using iterators. When iterators run out, the example restarts them (which should not happen with the fixed size client data) but showcases how this can be handled. The loss metrics are now aggregated by weighting them by the number of samples, and then averaged using the sum of the number of samples across all batches. This provides a more accurate result when client datasets have different sizes. This example illustrates the core concept of how to process datasets and use iterative data extraction, along with properly weighted metric aggregation.

Finally, let’s show an example where the model is updated between rounds. This is key to evaluate the effects of training on the evaluation performance.

```python
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def model_fn():
    # Define your Keras model here
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def make_client_data(num_clients, samples_per_client):
  # Simplified creation to have a fixed dataset size per client
  datasets = []
  for i in range(num_clients):
    features = tf.random.normal((samples_per_client, 10))
    labels = tf.random.uniform((samples_per_client, 1), maxval=2, dtype=tf.int32)
    datasets.append(tf.data.Dataset.from_tensor_slices((features, labels)).batch(32))
  return datasets


# Example usage
NUM_CLIENTS = 2
SAMPLES_PER_CLIENT = 100
ROUND_COUNT = 3
BATCH_SIZE = 32


# Create the federated evaluation function
evaluation = tff.learning.build_federated_evaluation(model_fn)

client_datasets = make_client_data(NUM_CLIENTS, SAMPLES_PER_CLIENT)


def train_and_evaluate_over_rounds(evaluation_function, client_data, num_rounds, batch_size):
    """Evaluates a model over multiple rounds using an iterator.

    Args:
        evaluation_function: A function returned by
            tff.learning.build_federated_evaluation.
        client_data: A list of tf.data.Datasets, each representing a client.
        num_rounds: The number of evaluation rounds to run.
    Returns:
        A dictionary of aggregated metrics.
    """

    model = model_fn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    total_metrics = {}
    num_examples = 0

    # Wrap the client data into iterators
    client_iterators = [iter(dataset) for dataset in client_data]

    for round_num in range(num_rounds):
      # Training phase (simplified for illustration)
      current_round_data = []
      for it in client_iterators:
          try:
            current_round_data.append(next(it))
          except StopIteration:
            # We ran out of samples from the dataset for this client, restart the iterator
            print("Restarting iterator, this should not happen in this example")
            client_iterators = [iter(dataset) for dataset in client_data]
            current_round_data.append(next(it)) # Retrieve the first batch again

      # Perform the training step
      for client_dataset in current_round_data:
        for features, labels in client_dataset:
          with tf.GradientTape() as tape:
            predictions = model(features)
            loss = loss_fn(labels, predictions)
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))



      # Evaluation phase
      federated_dataset = tff.simulation.datasets.TestClientData(current_round_data)
      round_metrics = evaluation_function(model, federated_dataset)

      # Aggregate metrics (summed, then averaged)
      for metric_name, metric_value in round_metrics.items():
        if metric_name not in total_metrics:
          total_metrics[metric_name] = 0.0
        total_metrics[metric_name] += metric_value * sum(np.prod(x.shape) for x in current_round_data[0][0])

      num_examples += sum(np.prod(x.shape) for x in current_round_data[0][0])
  
    # Average the metrics across all of the examples
    for metric_name, value in total_metrics.items():
        total_metrics[metric_name] = value/num_examples

    return total_metrics

# Perform the multi-round evaluation
aggregated_metrics = train_and_evaluate_over_rounds(evaluation, client_datasets, ROUND_COUNT, BATCH_SIZE)
print("Aggregated evaluation metrics:", aggregated_metrics)
```

In this third example, we have integrated a simplified training loop between the evaluation rounds. After extracting the next slice of data from the client datasets for the current round, a basic gradient descent optimization is applied using `tf.GradientTape`. This demonstrates how the evaluated model can evolve between evaluations, allowing you to assess the impact of training on the metrics.

For further exploration, I recommend reviewing the official TensorFlow Federated documentation, particularly the sections on federated evaluation and iterative processes. Additionally, examining the implementation of the `tff.learning.build_federated_averaging` function can provide insight into managing state and data across rounds in federated learning settings.  Studying examples involving iterative federated processes available in the examples from the TensorFlow Federated repository on Github is also very useful. Finally, the concepts of data iterators and their management in `tf.data` should be reviewed if data is required to be sampled.
