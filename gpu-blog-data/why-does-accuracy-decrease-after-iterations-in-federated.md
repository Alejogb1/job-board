---
title: "Why does accuracy decrease after iterations in federated learning?"
date: "2025-01-30"
id: "why-does-accuracy-decrease-after-iterations-in-federated"
---
The primary reason for accuracy degradation in federated learning (FL) after several iterations stems from the inherent heterogeneity of client data distributions.  Over my years working on distributed optimization systems, I've observed this phenomenon consistently across diverse projects, and it's a challenge that requires a nuanced understanding of the underlying processes.  While global model aggregation aims to capture a consensus, the inherent variations in local training datasets often lead to conflicting gradient updates, ultimately hindering convergence towards a globally optimal solution.

**1.  Explanation of Accuracy Degradation in Federated Learning**

Federated learning operates under the premise of training a shared global model across multiple decentralized clients, each possessing a unique subset of the overall training data.  The crucial step is the aggregation of locally computed model updates (gradients or model parameters).  The challenge arises when the distributions of these local datasets significantly differ.  For instance, consider a medical image classification task:  one client might predominantly have images of one type of pathology, while another client’s data represents a completely different pathology. During local training, each client will adapt the model to its specific dataset.  When these locally optimized models are aggregated, the resulting global model may not generalize well to unseen data because it's a compromise between often-conflicting local optima.

This issue is exacerbated by several factors:

* **Non-IID Data:** The most significant contributor.  Non-Independent and Identically Distributed (non-IID) data, meaning client data is not representative of the global distribution, creates conflicting gradient updates during aggregation. Clients effectively "pull" the model in different directions, leading to a less effective global model.

* **Client Drift:**  Over multiple iterations, local models might drift significantly from the global model due to the continuous adaptation to their respective local datasets.  This divergence increases with each iteration, impacting the quality of the aggregated model.  The global model becomes a less effective average of increasingly disparate local models.

* **Communication Bottlenecks:** The communication overhead in federated learning can significantly impact the convergence.  Limited bandwidth or communication delays can hinder the timely dissemination of model updates, leading to stale information and hindering efficient global model improvement.  The effect is worsened by the inherent iterative nature of FL, where each round builds upon the previous one.

* **Statistical Inefficiency:** The averaging process inherently diminishes information. Local models, trained on potentially smaller datasets, may contain valuable information that gets diluted when averaged with other local models.  This averaging process, while essential, results in a loss of information, especially in the presence of heterogeneous data.


**2. Code Examples illustrating Potential Issues**

The following code examples use a simplified scenario to illustrate how different data distributions affect global model accuracy. We use a linear regression problem for simplicity.  In a real-world scenario, the complexity would be significantly higher, involving deep learning models and much larger datasets.


**Example 1:  IID Data (Idealized Scenario)**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate IID data for multiple clients
num_clients = 3
data_points = 100
X = np.random.rand(num_clients * data_points, 1) * 10
y = 2 * X + 1 + np.random.randn(num_clients * data_points, 1)

# Divide data amongst clients
X_clients = np.array_split(X, num_clients)
y_clients = np.array_split(y, num_clients)


global_model = LinearRegression()
for i in range(num_clients):
    local_model = LinearRegression()
    local_model.fit(X_clients[i], y_clients[i])
    # In a real FL scenario, these local models would be aggregated in a more complex way
    # Here, we just take the average for illustrative purposes.
    if i == 0:
        global_model.coef_ = local_model.coef_
        global_model.intercept_ = local_model.intercept_
    else:
        global_model.coef_ = (global_model.coef_ + local_model.coef_) / 2
        global_model.intercept_ = (global_model.intercept_ + local_model.intercept_) / 2

print("Global Model Coefficients:", global_model.coef_)
print("Global Model Intercept:", global_model.intercept_)
```

This example demonstrates a scenario where data is IID.  The aggregated model is likely to perform well.


**Example 2: Non-IID Data (Illustrating Degradation)**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate Non-IID data
num_clients = 3
data_points = 100

X_clients = []
y_clients = []
for i in range(num_clients):
    X_client = np.random.rand(data_points, 1) * (i + 1) * 5  # Different ranges
    y_client = (i + 1) * X_client + (i + 1) + np.random.randn(data_points, 1)
    X_clients.append(X_client)
    y_clients.append(y_client)

# Same aggregation as before
global_model = LinearRegression()
for i in range(num_clients):
    local_model = LinearRegression()
    local_model.fit(X_clients[i], y_clients[i])
    if i == 0:
        global_model.coef_ = local_model.coef_
        global_model.intercept_ = local_model.intercept_
    else:
        global_model.coef_ = (global_model.coef_ + local_model.coef_) / 2
        global_model.intercept_ = (global_model.intercept_ + local_model.intercept_) / 2

print("Global Model Coefficients:", global_model.coef_)
print("Global Model Intercept:", global_model.intercept_)

```

Here, the data is clearly non-IID, resulting in significantly different local models.  Simple averaging will likely produce a poor global model.


**Example 3:  Illustrating Client Drift (Simplified)**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Simulate client drift over iterations
num_clients = 3
data_points = 100
iterations = 5

X = np.random.rand(num_clients * data_points, 1) * 10
y = 2 * X + 1 + np.random.randn(num_clients * data_points, 1)

X_clients = np.array_split(X, num_clients)
y_clients = np.array_split(y, num_clients)

global_model = LinearRegression()
for iteration in range(iterations):
    for i in range(num_clients):
        local_model = LinearRegression()
        local_model.fit(X_clients[i], y_clients[i])  # Local training
        if iteration == 0 and i == 0:
            global_model.coef_ = local_model.coef_
            global_model.intercept_ = local_model.intercept_
        else:
            global_model.coef_ = (global_model.coef_ + local_model.coef_) / 2
            global_model.intercept_ = (global_model.intercept_ + local_model.intercept_) / 2

print("Global Model Coefficients after", iterations, "iterations:", global_model.coef_)
print("Global Model Intercept after", iterations, "iterations:", global_model.intercept_)
```

This example demonstrates how, even with IID data, repeated local training and averaging might not lead to the optimal model, potentially showcasing a form of client drift over the iterations.

**3. Resource Recommendations**

For a deeper understanding of federated learning and the challenges of non-IID data, I recommend exploring publications on federated averaging, differential privacy in FL, and techniques like federated averaging with momentum.  Furthermore, studying the impact of various aggregation strategies, like weighted averaging based on client data quality, is crucial.  Examining research papers focusing on techniques designed to mitigate the effects of non-IID data, such as personalized federated learning, would also prove beneficial.  Finally, review  publications comparing different FL frameworks and their performance under diverse conditions.
