---
title: "Can SDGs be achieved with batch sizes greater than one?"
date: "2025-01-30"
id: "can-sdgs-be-achieved-with-batch-sizes-greater"
---
Achieving the Sustainable Development Goals (SDGs) hinges on scalability and efficiency, and the notion of whether batch sizes exceeding one are conducive to this is critical. From my experience optimizing large-scale machine learning models for resource allocation, specifically in agriculture, I've observed that while single-instance, or 'batch size one', approaches offer a degree of fine-grained control, they are fundamentally impractical for the scale required to meaningfully impact the SDGs. Larger batch sizes, while introducing some challenges, are essential for efficient resource utilization and ultimately achieving measurable progress towards these global objectives.

The core issue revolves around the inherent limitations of processing individual units when dealing with vast datasets and widespread deployment. Consider the problem of predicting crop yields to optimize fertilizer distribution. If we were to model each field separately (batch size one), we would be leveraging only data specific to that field to update our models. This significantly slows down the learning process and renders the model less robust to variance between fields. In addition, this is operationally costly as the overhead associated with updating the model per field becomes a burden. Furthermore, the potential for parallel processing is severely hampered. A larger batch size, conversely, allows the model to learn from the statistical distribution present across multiple fields simultaneously, leading to more accurate and reliable predictions while amortizing the computational cost.

My experience with deep learning models has consistently demonstrated this principle. Specifically, using a stochastic gradient descent approach, each model update using a batch size one relies heavily on the gradient computed from a single data point. This introduces significant noise in the update and impedes model convergence. Furthermore, this methodology severely underutilizes the processing power offered by modern hardware, which is typically designed to work with batches. The larger batch sizes allow for better approximation of the overall gradient, reducing the noise and enabling faster and more stable convergence.

To further clarify, I can outline several examples where batch sizes greater than one have been instrumental in achieving results pertinent to SDGs.

**Example 1: Optimizing Water Usage in Irrigation**

Imagine a sensor network monitoring soil moisture levels across a large agricultural cooperative. We want to develop a predictive model that determines optimal irrigation schedules for different regions within the cooperative.

```python
import numpy as np
import tensorflow as tf

# Assume 'soil_data' is a numpy array of shape (N, M)
# where N is number of data points and M is number of features (e.g. soil moisture, temperature, rainfall)
# assume 'irrigation_needs' is a numpy array of shape (N,)
# which represents optimal irrigation for each corresponding soil_data instance
# Assume the dataset is already created and loaded into these two variables
batch_size = 64  # Use batch size of 64 for training
dataset = tf.data.Dataset.from_tensor_slices((soil_data, irrigation_needs))
dataset = dataset.batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(soil_data.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer predicts irrigation need
])

model.compile(optimizer='adam', loss='mse')

for epoch in range(10): # Training for 10 epochs
    for batch_soil, batch_irrigation in dataset:
        model.train_on_batch(batch_soil, batch_irrigation)

```

In this example, we load our dataset and then use TensorFlow to create a batch dataset object. We train our model using a batch size of 64. The `train_on_batch` function performs a gradient descent update on the model based on this batch of training data. Increasing the batch size improves computational efficiency by vectorizing the calculations. Importantly, the model learns from diverse examples within each batch, improving generalizability to different agricultural regions within the cooperative and therefore reducing water waste at scale, a crucial step towards SDG 6 (Clean Water and Sanitation). The alternative of batch size one would require 64 individual calls to `train_on_batch`, which would make the training process slower.

**Example 2: Predicting Energy Consumption for Rural Electrification**

Consider a project focused on implementing smart grids in rural areas, which would help achieve SDG 7 (Affordable and Clean Energy). We collect data on household energy consumption along with various features such as household size, appliance type, weather conditions, and time of day.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Assume 'energy_data' is a pandas dataframe with energy consumption
# as target variable and relevant features
# Assume the dataset is already created and loaded into the variable

batch_size = 256 # Using a larger batch size for a different model type

X = energy_data.drop('energy_consumption', axis=1)
y = energy_data['energy_consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Batched training through iterative training loops, mimicking larger batch sizes
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_batch, y_batch)

```

In this scenario, instead of using a neural network, we're implementing a Random Forest Regressor. Random Forests don’t explicitly have batching in their `fit` method, but we approximate batched training by iterating through the training set with a large batch size. While this does not leverage the same parallelization potential as neural network batching, it still allows for efficient memory utilization compared to iterative training. This model, trained on a batch of energy data, can predict energy consumption patterns, enabling effective resource planning and grid load balancing to reduce energy waste and increase access to sustainable energy in rural areas. A model trained with a batch size of one would generalize poorly due to a lack of statistical diversity.

**Example 3: Optimizing Supply Chain Logistics for Food Distribution**

Efficient supply chains are paramount to achieving SDG 2 (Zero Hunger). We may have data on food production, storage capacity, transportation routes, and demand patterns. Our objective is to optimize these logistical networks to minimize waste and ensure timely food delivery.

```python
import numpy as np
from scipy.optimize import minimize

# Assume 'supply_data' and 'demand_data' are numpy arrays representing supply and demand
# for a series of locations
# Assume that 'distances' represents the distances between the locations

def logistic_loss(x, supply_data, demand_data, distances):
    # Simple loss function that penalizes imbalance between supply and demand
    # and penalizes transport distance
    transport_matrix = x.reshape(supply_data.shape[0], demand_data.shape[0])
    supply_satisfied = np.sum(transport_matrix, axis=1)
    demand_satisfied = np.sum(transport_matrix, axis=0)
    imbalance = np.sum((supply_data - supply_satisfied)**2) + np.sum((demand_data - demand_satisfied)**2)
    transport_cost = np.sum(distances * transport_matrix)
    return imbalance + transport_cost

# Initial guess of transport matrix
init_transport = np.ones(supply_data.shape[0] * demand_data.shape[0])
bounds = [(0, None) for _ in range(len(init_transport))] # Non-negative transport

result = minimize(logistic_loss, init_transport, args=(supply_data, demand_data, distances),
                        method='L-BFGS-B', bounds=bounds) # Minimization of the loss function

optimal_transport = result.x.reshape(supply_data.shape[0], demand_data.shape[0]) # Reshape solution
```

This example utilizes a numerical optimization algorithm (SciPy's `minimize`) instead of training a model explicitly. While batching is not used directly in `minimize`, the concept is similar:  we are not optimizing the system for each supply/demand pair separately. Instead, the `logistic_loss` function considers the system's state as a whole – the total supply, the total demand, and all potential transport routes at once. A batch-size-one approach would require optimizing each pair individually, missing inter-dependencies and systemic inefficiencies. The result of this optimization is an optimal transport matrix that minimizes both waste and transportation costs. This type of optimization, which considers all locations simultaneously, greatly assists in achieving SDG 2 by efficiently distributing food resources.

For further understanding of this topic, I recommend studying materials on the following: Stochastic gradient descent optimization techniques, specifically how batch size affects training dynamics; parallel processing and vectorization principles related to machine learning computation; and concepts related to resource optimization and supply chain management within operations research. These areas offer a strong theoretical foundation and provide practical insights into the benefits of using batch sizes greater than one in the context of achieving SDGs. Furthermore, understanding how these principles apply to various types of models and optimization tasks provides a deeper insight into the underlying mechanics that affect the performance of each implementation. The idea of considering inter-dependencies and system-wide optimizations versus independent sub-optimizations is key to efficiently tackling the complexities of SDGs.
