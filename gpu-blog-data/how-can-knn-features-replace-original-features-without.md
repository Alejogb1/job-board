---
title: "How can kNN features replace original features without iteration?"
date: "2025-01-30"
id: "how-can-knn-features-replace-original-features-without"
---
The challenge of integrating k-Nearest Neighbors (kNN) features as a direct replacement for original features, bypassing iterative processes, necessitates a nuanced understanding of how kNN transformations can be precomputed and subsequently used as a novel representation of the data. My experience, while working on real-time sensor data processing for a manufacturing line, involved precisely this challenge, where traditional, iterative methods proved too computationally expensive for timely analysis. The core idea lies in leveraging the concept of neighborhood representations that can be generated once and then plugged directly into downstream models, rather than recomputing neighborhoods during each model training or inference cycle.

A kNN transformation effectively re-encodes a data point based on its proximity to other points in the dataset. Instead of retaining the original feature values, a new feature vector is constructed by considering the labels of its *k* nearest neighbors. This approach generates new features with different characteristics compared to the original input features and reduces the dimensionality of the data for simpler, more effective modelling. This transformation is generally an unsupervised method, relying on the similarity between input vectors to select the neighbors, and is often used for creating new, richer feature sets. Specifically, we're focusing on how this transformation can be done in a pre-processing stage, eliminating the need for dynamic neighborhood calculations during modeling. The critical factor is to build these features once and then use them directly.

To illustrate, consider a dataset with three features – X, Y, and Z – representing, say, the dimensions of a machined part. The goal is to replace these original XYZ features with kNN-derived features. Instead of using raw XYZ values, we will create new features based on the dominant label amongst the *k* nearest neighbors of each data point. This means that for each data point, we identify its k nearest neighbors in the original feature space (XYZ space in this case). Once those neighbors have been identified, we can assign labels based on the labels of those neighbors. Depending on the method, the new representation can then be used as a replacement for or an augmentation of the original features.

Here's how this can be achieved with an example using Python with NumPy and SciKit-learn, along with detailed explanations of each step:

**Code Example 1: kNN Based Class Labels**

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# Sample data with original features and labels
X = np.array([[1, 2, 3], [1.5, 1.8, 2.9], [5, 8, 6], [8, 8, 5], [1, 0.6, 2], [9, 1, 1]])
y = np.array(['A', 'A', 'B', 'B', 'A', 'C'])

# Encode labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Define the number of neighbors
k = 3

# Train a NearestNeighbors model on original data
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(X)

# Find the k nearest neighbor indices for each data point
distances, indices = knn_model.kneighbors(X)


# Aggregate the neighbor labels for each data point, take mode
knn_labels = np.zeros(len(X), dtype=int)
for i in range(len(indices)):
    neighbor_labels = y_encoded[indices[i]]
    counts = np.bincount(neighbor_labels)
    knn_labels[i] = np.argmax(counts)

# Decode the labels back to original form
y_knn_transformed = label_encoder.inverse_transform(knn_labels)

print("Original Labels:\n", y)
print("\nkNN Transformed Labels:\n", y_knn_transformed)
```

This code first creates a NearestNeighbors model based on the original features. Then, using the 'kneighbors' method, the code obtains, for each point in the feature space, an array of *k* nearest neighbor indices. A new label for each data point is calculated based on the most frequent label amongst these *k* nearest neighbors. Finally, the transformed label set can be used downstream in modeling tasks. Note that instead of directly obtaining labels, the *distances* to the k neighbors and the *average* of k neighbors can also be used as an alternative feature set, as demonstrated in the next example.

**Code Example 2:  kNN Based Distance Features**

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Sample data with original features
X = np.array([[1, 2, 3], [1.5, 1.8, 2.9], [5, 8, 6], [8, 8, 5], [1, 0.6, 2], [9, 1, 1]])

# Define the number of neighbors
k = 3

# Train a NearestNeighbors model on original data
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(X)

# Find the k nearest neighbor distances for each data point
distances, indices = knn_model.kneighbors(X)

# For each data point, take average distance to its k neighbors
knn_distances = np.mean(distances, axis = 1)

print("Original features:\n", X)
print("\nkNN Transformed Distance Feature:\n", knn_distances)
```

In this example, after finding the k nearest neighbors, we instead generate a new feature representing the *average distance* to the *k* neighbors for each data point. This yields a single numerical feature for every data point instead of a label. The distance metric used for determining neighbors is, by default, the Euclidean distance. This can be customized based on the nature of the original features using the `metric` parameter when instantiating the `NearestNeighbors` model. This results in a potentially more nuanced representation than just the dominant neighbor label.

**Code Example 3: Combining Original and kNN Distance Features**

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Sample data with original features
X = np.array([[1, 2, 3], [1.5, 1.8, 2.9], [5, 8, 6], [8, 8, 5], [1, 0.6, 2], [9, 1, 1]])

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Define the number of neighbors
k = 3

# Train a NearestNeighbors model on original data
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(X_scaled)


# Find the k nearest neighbor distances for each data point
distances, indices = knn_model.kneighbors(X_scaled)


# Average distances to create a new feature
knn_distances = np.mean(distances, axis=1)

# Add kNN distance feature to original features
X_knn_combined = np.column_stack((X, knn_distances))

print("Original features:\n", X)
print("\nCombined Features with kNN distance:\n", X_knn_combined)
```

This third example illustrates that instead of directly replacing the original features, we can augment them by creating kNN features and combining them with the original feature set. Here, prior to creating the `NearestNeighbors` model, I apply standardization to the dataset, which normalizes all features to zero mean and unit variance. This is an important step when distances are used in algorithms as it prevents features with larger values from dominating the kNN process. Then, I calculate the average distance of each point to its *k* neighbors and append it as a new feature column to the original feature matrix. This results in a dataset where both original feature information and kNN-derived neighborhood information are available, potentially improving the performance of downstream models.

These examples highlight the flexibility in how kNN features can be derived and subsequently used. The key takeaway is that the kNN neighborhood search and feature generation are done *once* as a pre-processing step. The transformed features—whether based on neighbor labels, distances, or some other aggregation—are then static and can be directly fed into any downstream model. This approach removes the need for iterative kNN computations during model training or inference, leading to significant performance gains, especially for large datasets or time-critical applications like the industrial sensor data I encountered.

For deeper exploration, resources focusing on unsupervised learning and feature engineering are particularly valuable. Books covering machine learning algorithms, such as those emphasizing practical implementation with SciKit-learn, are recommended. Additionally, exploring the official documentation for SciKit-learn’s `NearestNeighbors` class offers comprehensive coverage of available parameters and customization options. Furthermore, academic papers focused on dimensionality reduction techniques and feature generation with neighborhood-based methods can provide deeper theoretical insights. In particular, papers that explore how neighbor-based features can be used for image recognition or time-series analysis often provide helpful guidance for how such methods are applied in practice.
