---
title: "How can data be split for training and testing in federated learning?"
date: "2025-01-30"
id: "how-can-data-be-split-for-training-and"
---
Federated learning inherently poses a unique challenge when splitting data for training and testing because the data is distributed across numerous devices or silos, and direct access to a centralized dataset is typically unavailable. My experience developing federated learning solutions for healthcare systems has demonstrated that the traditional hold-out method used in centralized machine learning is not directly applicable. We must tailor our strategies to handle the decentralized nature of the data. This requires careful consideration of data heterogeneity and distribution imbalances among participating clients.

The primary distinction in federated learning data splits lies in how we conceptualize the 'test set.' In a conventional setting, a test set is carved from the overall available data before model training commences. In federated learning, we often lack this luxury. Instead, we are dealing with data *already* distributed amongst devices, and generating a shared, centralized test set would violate the core privacy premise of the methodology. Instead, we have to implement a distributed testing paradigm. Each client maintains its local testing dataset, which might not be representative of the overall population but is reflective of its own local data distribution.

I’ve found that the process generally involves two main stages: defining client-local splits and then orchestrating the distributed test phase. For client-local splits, the data residing on each client device is divided into training and testing subsets, mimicking traditional splitting. The ratio here often mirrors what’s seen in centralized machine learning scenarios, for example, 80% training and 20% testing. However, these ratios can be adjusted on a per-client basis if the data availability varies greatly among devices. Data shuffling on each client before splitting is essential to minimize bias due to any existing order within the local data.

Once the client-local splits are established, the distributed testing is initiated, ideally after federated model training. The global model is pushed to all participating clients, and each client evaluates the model on its respective local testing dataset. Client-local testing results are aggregated (usually averaged) on the central server to estimate overall model performance. This approach avoids needing a shared centralized test dataset, maintaining data privacy and adhering to the federated learning's core principles. These results are utilized for subsequent refinement and comparison of model variations.

Here are three practical examples illustrating data splitting with accompanying commentary:

**Example 1: Simple Random Split Within a Client (Python)**

```python
import numpy as np

def client_data_split(data, train_ratio=0.8):
  """Splits client's data into training and testing subsets.

  Args:
    data: A numpy array representing the client's local dataset.
    train_ratio: The ratio of data to be used for training.

  Returns:
      A tuple containing (train_data, test_data).
  """
  np.random.shuffle(data) # Ensure no inherent ordering bias
  split_index = int(len(data) * train_ratio)
  train_data = data[:split_index]
  test_data = data[split_index:]
  return train_data, test_data

# Example usage (simulated data):
client_data = np.array([i for i in range(100)])
train_data, test_data = client_data_split(client_data)
print(f"Training data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}")
```

*Commentary:* This example shows how a simple random split can be achieved. The data is shuffled before splitting to prevent any bias introduced due to the initial ordering of the data. `train_ratio` allows flexibility in adjusting the proportions of data allotted to the training and testing sets based on specific client requirements. This function assumes that the client's local data is representable as a NumPy array, which is often the case when processing numerical data or vectors.

**Example 2: Stratified Split Based on Labels (Python)**

```python
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_client_split(features, labels, train_ratio=0.8):
    """Splits client's data into training and testing with stratification.

    Args:
        features: Numpy array of features.
        labels: Numpy array of corresponding labels.
        train_ratio: The ratio of data to be used for training.

    Returns:
       A tuple of (train_features, test_features, train_labels, test_labels)
    """
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=1-train_ratio, stratify=labels, random_state=42)
    return train_features, test_features, train_labels, test_labels


# Example usage:
features = np.random.rand(100, 10)  # 100 samples, 10 features
labels = np.random.randint(0, 3, 100) # 3 classes
train_features, test_features, train_labels, test_labels = stratified_client_split(features, labels)
print(f"Training features shape: {train_features.shape}")
print(f"Testing features shape: {test_features.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing labels shape: {test_labels.shape}")

```

*Commentary:* This code implements a stratified split using Scikit-learn's `train_test_split` function, useful when dealing with imbalanced class distributions within a client's data. By setting the `stratify` parameter equal to the `labels` array, we ensure that the training and testing subsets maintain the same proportions of each class as found in the original dataset. The `random_state` ensures reproducibility for consistent debugging. This is crucial for scenarios when class representation affects the performance of a machine learning model.

**Example 3: Handling Client Data of Varying Sizes (Pseudocode)**

```python
# Pseudocode demonstrating client-specific split ratios

#Assume each client has a dictionary of data: {'features': numpy array, 'labels': numpy array}

client_data = {
    'client1': {'features': np.random.rand(100,5), 'labels': np.random.randint(0, 2, 100)},
    'client2': {'features': np.random.rand(500,5), 'labels': np.random.randint(0, 2, 500)},
    'client3': {'features': np.random.rand(200,5), 'labels': np.random.randint(0, 2, 200)}
}

train_data_per_client = {}
test_data_per_client = {}

for client_id, data in client_data.items():
    features = data['features']
    labels = data['labels']
    if client_id == 'client1':
      train_ratio = 0.7 # Assign specific train ratio if needed
    elif client_id == 'client2':
      train_ratio = 0.9
    else:
       train_ratio = 0.8

    train_features, test_features, train_labels, test_labels = stratified_client_split(features, labels, train_ratio)
    train_data_per_client[client_id] = {'features':train_features, 'labels':train_labels}
    test_data_per_client[client_id] = {'features':test_features, 'labels':test_labels}


for client_id, train_data in train_data_per_client.items():
     print(f"Training data size for {client_id}: {train_data['features'].shape}")

for client_id, test_data in test_data_per_client.items():
    print(f"Testing data size for {client_id}: {test_data['features'].shape}")
```

*Commentary:* This example uses pseudocode to illustrate a scenario where clients have differing amounts of data. It shows how we could assign different training-to-testing ratios to each client's data based on its available quantity. The iterative structure demonstrates how the same function (`stratified_client_split`) can be adapted to diverse client scenarios. The `if` statements allow for client-specific training ratios, thereby addressing potential data imbalance issues across the federation. This kind of individual adjustment improves the robustness of federated learning solutions.

For further information, I recommend delving into works that discuss federated learning strategies and data distribution. Specifically, materials detailing the various challenges of data heterogeneity and how these challenges can be addressed using careful data split design and algorithms that can handle data imbalances. It's also worth exploring resources that discuss concepts of privacy-preserving machine learning, which provides a broader context for distributed evaluation methods.
