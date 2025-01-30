---
title: "Why is model performance not improving during federated learning training?"
date: "2025-01-30"
id: "why-is-model-performance-not-improving-during-federated"
---
Federated learning's inherent data heterogeneity often manifests as a significant obstacle to model performance improvement.  My experience working on a large-scale medical image classification project highlighted this acutely.  While the distributed nature offered privacy advantages, the variation in imaging techniques and annotation practices across participating hospitals consistently hampered convergence to a high-performing global model. This lack of improvement stems from several interwoven factors, which I will address below.

**1. Data Heterogeneity and its Impact:**

The core challenge in federated learning lies in the non-identical data distributions across participating clients.  Each client possesses a local dataset reflecting its unique environment, leading to discrepancies in feature distributions, class imbalances, and even differing label interpretations.  This heterogeneity directly impacts model training. During local updates, models adapt to their specific datasets, leading to client-specific model variations. When these locally trained models are aggregated to form a global model, conflicting updates can effectively cancel each other out, preventing convergence towards a globally optimal solution.  The aggregation process, even using weighted averaging techniques, struggles to reconcile these conflicting local updates, resulting in a stagnant or slowly improving global model.  This was directly observed in my project; hospitals using older MRI scanners produced significantly different feature distributions compared to those with newer technology, severely impacting the global model's accuracy on unseen data.

**2. Communication Bottlenecks and Limited Data Exchange:**

Federated learning necessitates communication between the central server and the participating clients.  This communication overhead can become a significant limiting factor.  Bandwidth constraints, especially in resource-limited environments, can restrict the frequency and volume of model updates exchanged.  Insufficient communication leads to slower convergence, as the global model receives infrequent updates, and the individual client models may not benefit from sufficient global information.  In my project, limited bandwidth in one participating hospital resulted in infrequent updates, leading to that hospital's model lagging behind, and negatively affecting the overall global model accuracy.  Moreover, the restricted exchange of data – a fundamental aspect of preserving privacy – means the global model is never exposed to the complete dataset, thus limiting its capacity to learn optimal representations.

**3. Client Drift and Model Divergence:**

The decentralized nature of federated learning allows local models to diverge significantly from the global model over time. This "client drift" is exacerbated by the heterogeneous data.  As local models adapt to their specific datasets, they may develop unique features and biases, rendering the aggregated global model less effective.  This divergence can lead to a situation where the global model is not representative of the overall data distribution, thereby hindering performance improvement.  We mitigated some of this in our medical image project by implementing a stricter model architecture standardization across clients, reducing the degree of possible divergence, although data heterogeneity remained a limiting factor.

**4. Inadequate Model Architecture and Hyperparameter Tuning:**

The choice of model architecture and hyperparameters significantly influence performance in any machine learning task; federated learning is no exception.  An inappropriately chosen architecture may be unable to effectively capture the variations in data across clients.  Similarly, poorly tuned hyperparameters can lead to slow convergence or poor generalization.  In our project, we initially used a relatively simple convolutional neural network (CNN) for image classification.  The transition to a more complex architecture with transfer learning significantly improved the performance, especially after we adapted hyperparameters such as learning rate and batch size using techniques like Bayesian Optimization tailored to the federated setting.


**Code Examples and Commentary:**

The following examples illustrate different aspects of federated learning and the challenges related to performance improvement.  These are simplified illustrations, and practical implementations would require more robust frameworks and error handling.

**Example 1:  Simple Federated Averaging (FedAvg)**

```python
import numpy as np

# Simulate client data (simplified)
client_data = [np.random.rand(100, 10), np.random.rand(150, 10), np.random.rand(200, 10)]
client_labels = [np.random.randint(0, 2, 100), np.random.randint(0, 2, 150), np.random.randint(0, 2, 200)]

# Simulate local training (simplified) – replace with actual model training
def local_training(data, labels):
    # Simulate model update – replace with actual model update
    return np.random.rand(10)

# Federated Averaging
global_model = np.zeros(10)
num_clients = len(client_data)
for i in range(10): # Number of rounds
    client_updates = []
    for j in range(num_clients):
        update = local_training(client_data[j], client_labels[j])
        client_updates.append(update)
    global_model = np.mean(client_updates, axis=0)
    print(f"Global model after round {i+1}: {global_model}")
```

This example demonstrates a basic FedAvg algorithm.  Note the simplification; actual model training and update calculations would be significantly more complex.  The performance limitations of this method stem from the simplistic nature of the `local_training` function and the absence of mechanisms to address data heterogeneity.

**Example 2: Addressing Data Heterogeneity with Weighted Averaging**

```python
import numpy as np

# ... (client_data and client_labels as before) ...

# Weighted Averaging based on dataset size
weights = [len(data) for data in client_data]
total_samples = sum(weights)
normalized_weights = [w / total_samples for w in weights]

# Federated Averaging with weights
global_model = np.zeros(10)
for i in range(10):
    client_updates = []
    for j in range(num_clients):
        update = local_training(client_data[j], client_labels[j])
        client_updates.append(update)
    global_model = np.average(client_updates, axis=0, weights=normalized_weights)
    print(f"Global model after round {i+1}: {global_model}")
```

This example incorporates weighted averaging based on the size of each client's dataset. While this partially mitigates the effect of dataset size differences, it does not address other forms of heterogeneity, such as differing feature distributions or label noise.

**Example 3: Incorporating Differential Privacy**

```python
import numpy as np

# ... (client_data and client_labels as before) ...

# Add simple differential privacy to local updates (highly simplified)
def add_noise(update, epsilon):
    noise = np.random.laplace(0, 1/epsilon, size=len(update))
    return update + noise

# Federated Averaging with Differential Privacy
global_model = np.zeros(10)
epsilon = 1.0 # Privacy parameter
for i in range(10):
    client_updates = []
    for j in range(num_clients):
        update = local_training(client_data[j], client_labels[j])
        noisy_update = add_noise(update, epsilon)
        client_updates.append(noisy_update)
    global_model = np.mean(client_updates, axis=0)
    print(f"Global model after round {i+1}: {global_model}")

```

This example adds differential privacy to the local updates.  The `epsilon` parameter controls the trade-off between privacy and accuracy.  Note that this is a highly simplified example; robust differential privacy mechanisms are significantly more complex.  The noise added to protect privacy will inevitably negatively impact model accuracy.

**Resource Recommendations:**

For a deeper understanding of Federated Learning, consult relevant textbooks on distributed machine learning, and research papers on FedAvg, Federated Averaging with differential privacy, and techniques addressing data heterogeneity like personalized federated learning.  Explore the theoretical underpinnings of convergence guarantees and the impact of various aggregation methods.  Examine the literature on communication-efficient federated learning algorithms.  Finally, delve into practical implementations through open-source libraries commonly used in federated learning.
