---
title: "How can custom accuracy metrics be implemented in TensorFlow Federated?"
date: "2025-01-30"
id: "how-can-custom-accuracy-metrics-be-implemented-in"
---
TensorFlow Federated (TFF) presents unique challenges when implementing custom accuracy metrics, primarily stemming from its decentralized nature and the inherent complexities of aggregating results across diverse client devices.  My experience working on a privacy-preserving healthcare application underscored this:  the need to compute a weighted F1-score across geographically distributed hospitals, each with varying patient populations, required a tailored approach beyond TFF's built-in metrics.

The core issue lies in the distinction between *local* computations performed on individual client devices and *global* aggregation managed by the federated server.  Custom metrics necessitate careful design to ensure both correctness and efficiency in this distributed environment.  Standard TensorFlow methods, suitable for centralized training, often fail to translate directly due to limitations in data transfer and computational resources available on client devices.  Therefore, implementing a custom metric involves strategically defining both the local computation and the aggregation strategy.

**1.  Clear Explanation:**

The implementation of custom metrics in TFF generally follows these steps:

* **Define the local computation:** This step involves writing a function that calculates the metric on a single client's data.  This function must operate on a `tff.Computation` that operates on the client's local dataset, typically a `tf.data.Dataset` or a similar structure.  The output of this function should be a simple tensor (or a structured set of tensors) representing the metric's value for that client.  Crucially, this computation must be self-contained and avoid dependencies on global data.

* **Define the aggregation function:**  This function, also a `tff.Computation`, receives the metric values from all clients and combines them to produce a global metric.  The choice of aggregation depends heavily on the metric itself.  Simple metrics like accuracy might use a weighted average, whereas more complex metrics may necessitate more sophisticated aggregation strategies.

* **Integrate into the TFF training loop:** Finally, the custom metric's local computation is incorporated within the TFF `tff.federated_compute` or `tff.federated_eval` mechanisms, ensuring it's executed on the clients concurrently.  The resulting metric values are then fed into the aggregation function. This often involves modifying a pre-existing TFF training loop or building a custom one.

This process requires a deep understanding of TensorFlow's data structures and TFF's federated computation primitives.  The following code examples illustrate these concepts.

**2. Code Examples with Commentary:**

**Example 1: Federated Accuracy**

This example demonstrates a straightforward federated accuracy calculation.  We assume a classification task.

```python
import tensorflow as tf
import tensorflow_federated as tff

def client_accuracy(local_data, model):
  # local_data: tf.data.Dataset of (example, label) pairs
  # model: tf.keras.Model
  predictions = model.predict(local_data.map(lambda x, y: x))
  labels = local_data.map(lambda x, y: y)
  accuracy_metric = tf.keras.metrics.Accuracy()
  accuracy_metric.update_state(labels, predictions)
  return accuracy_metric.result()

@tff.federated_computation(tff.type_at_clients(tf.float32), tff.type_at_server(tf.keras.Model))
def federated_accuracy(client_data, global_model):
  # client_data: a list of tf.data.Dataset objects, one per client
  # global_model: A shared Keras model
  return tff.federated_mean(
      tff.federated_map(client_accuracy, [client_data, global_model])
  )

# Example usage (simplified for brevity)
# ... (Assume 'federated_train' function already exists and trains the model) ...
client_data =  # ... client data, suitably structured ...
global_model = federated_train(client_data)
accuracy = federated_accuracy(client_data, global_model)
print(f"Federated Accuracy: {accuracy}")
```

This code defines `client_accuracy`, which computes accuracy on each client using TensorFlow's built-in metrics.  `federated_accuracy` then averages these individual client accuracies across the federation using `tff.federated_mean`.


**Example 2: Federated Weighted F1-Score**

This example showcases a more complex scenario—calculating a weighted F1-score.  This requires custom aggregation beyond simple averaging.

```python
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def client_weighted_f1(local_data, model, num_classes):
    #... (logic to compute precision, recall, and weighted F1-score for each client)...
    #Returns a tuple (precision, recall, f1)

@tff.federated_computation(tff.type_at_clients(tf.float32), tff.type_at_server(tf.keras.Model), tff.type_at_server(tf.int32))
def federated_weighted_f1(client_data, global_model, num_classes):
    client_metrics = tff.federated_map(lambda client_data, model, num_classes: client_weighted_f1(client_data, model, num_classes), [client_data, global_model, num_classes])
    #Aggregate precision, recall and f1 scores across clients using weighted average based on client data size
    # ... (Complex aggregation logic using tff.federated_sum and tff.federated_mean, potentially handling missing values)...
    # Returns a tuple (global precision, global recall, global F1-score)
```

This example highlights the need for more intricate aggregation in `federated_weighted_f1`.  The aggregation would likely involve summing weighted precisions, recalls, and subsequently computing the global weighted F1-score based on these sums and client dataset sizes.  Error handling for potential discrepancies in client data structures is crucial here.

**Example 3:  Federated AUC (Area Under the Curve)**

AUC calculation requires specific handling due to its reliance on ranking predictions.

```python
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.metrics import roc_auc_score

def client_auc(local_data, model):
    # Assume local_data is a tf.data.Dataset with (features, labels)
    y_true = local_data.map(lambda x, y: y).numpy()
    y_scores = model.predict(local_data.map(lambda x, y: x)).numpy()
    return roc_auc_score(y_true, y_scores)

@tff.federated_computation(tff.type_at_clients(tf.float32), tff.type_at_server(tf.keras.Model))
def federated_auc(client_data, global_model):
    client_aucs = tff.federated_map(lambda client_data, model: client_auc(client_data, model), [client_data, global_model])
    #This uses a weighted average based on number of samples in each client dataset
    weighted_average_auc = tff.federated_mean(client_aucs)
    return weighted_average_auc

```

This example uses `roc_auc_score` from scikit-learn. It assumes clients have enough data for a reliable AUC calculation.  A more robust implementation might incorporate checks for sufficient data or alternative metrics in case of insufficient samples on a client.


**3. Resource Recommendations:**

The official TensorFlow Federated documentation.  Thorough study of the `tff.federated_compute` and `tff.federated_eval` APIs is essential.  Understanding the nuances of different federated aggregation operators within TFF is also paramount.  Familiarity with TensorFlow's data handling capabilities and the intricacies of working with `tf.data.Dataset` objects will significantly enhance your ability to implement custom metrics efficiently.  Explore resources focused on distributed machine learning and federated learning algorithms to gain a deeper understanding of the underlying principles.
