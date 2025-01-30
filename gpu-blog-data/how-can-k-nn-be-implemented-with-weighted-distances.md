---
title: "How can k-NN be implemented with weighted distances?"
date: "2025-01-30"
id: "how-can-k-nn-be-implemented-with-weighted-distances"
---
The efficacy of a k-Nearest Neighbors (k-NN) classifier hinges significantly on its distance calculation. Standard implementations often assume all neighbors contribute equally to a classification decision. However, this fails to account for the varying degrees of relevance between a query point and its neighbors. Incorporating weighted distances, where closer neighbors have a stronger influence, can dramatically improve accuracy, especially in datasets with non-uniform distributions or noisy features. In my experience developing a predictive model for customer churn, the naive k-NN approach performed poorly until I implemented weighted distances which yielded a significant increase in precision.

The core principle of weighted k-NN involves assigning weights to each neighbor based on its proximity to the query point. Instead of a simple majority vote among the *k* neighbors, each neighbor's vote is modulated by its weight. This introduces a continuous influence scale, allowing nearby neighbors to have a more pronounced impact on the final classification. The specific formula for calculating these weights is flexible and can be adjusted based on the dataset characteristics and the desired outcome.

Several weighting functions can be employed. The inverse distance weighting (IDW) is common, where the weight assigned to a neighbor is inversely proportional to its distance from the query point. A simple IDW formula could be weight = 1 / distance. More complex functions, such as Gaussian weighting, utilize a kernel function to assign weights based on a normal distribution centered around the query point. These can provide more nuanced influence. Another effective function is exponential decay where weight = exp(-distance/bandwidth). Here bandwidth is a parameter that controls the spread.

Implementing weighted k-NN requires some changes from the standard approach. After computing the distances to the *k* nearest neighbors, their corresponding weights must be computed based on a selected formula. For classification, the votes for each class should be multiplied by the weights. The class with the highest total weighted votes is chosen as the predicted class. For regression tasks, the predicted value is a weighted average of the neighbor values.

Consider a classification scenario using a custom weighted distance calculation. I will focus on implementation using Python. The primary libraries needed are NumPy for numerical computation and SciPy for optimized distance functions.

```python
import numpy as np
from scipy.spatial import distance
from collections import defaultdict

def weighted_knn_classify(train_data, train_labels, test_point, k, weighting_function):
    """
    Classifies a test point using weighted k-NN.

    Args:
        train_data (np.ndarray): Training data points (n_samples, n_features)
        train_labels (np.ndarray): Labels for the training data (n_samples,)
        test_point (np.ndarray): Test data point (n_features,)
        k (int): Number of neighbors to consider.
        weighting_function (function): Function to compute the weight for distance.

    Returns:
        predicted_label (int): The predicted class label for the test point.
    """

    distances = [distance.euclidean(test_point, train_point) for train_point in train_data]
    #Sort the distances and extract k nearest distances and corresponding indices
    k_nearest_indices = np.argsort(distances)[:k]

    k_nearest_distances = np.array(distances)[k_nearest_indices]
    k_nearest_labels = train_labels[k_nearest_indices]

    weighted_votes = defaultdict(float)
    for i, dist in enumerate(k_nearest_distances):
         weight = weighting_function(dist)
         weighted_votes[k_nearest_labels[i]] += weight
    
    # Return the label with max weighted votes
    return max(weighted_votes, key=weighted_votes.get)
#Example usage
if __name__ == '__main__':
    # Generate some dummy training data
    train_data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9,11]])
    train_labels = np.array([0, 0, 1, 1, 0, 1])
    test_point = np.array([2, 2])
    k = 3

    # Implement inverse distance weighting
    def inverse_distance_weighting(dist):
        if dist == 0:
            return 1 #Avoid division by zero
        return 1 / dist

    predicted_label = weighted_knn_classify(train_data, train_labels, test_point, k, inverse_distance_weighting)
    print(f"Test point classified as {predicted_label} using inverse distance weighting")
```

In this example, I have defined a basic `weighted_knn_classify` function. The distances are calculated using `scipy.spatial.distance.euclidean`. I have defined the `inverse_distance_weighting` function, ensuring a return of 1 if the distance is zero. The function then computes the weighted vote for each class and return the class with the maximum votes. This snippet illustrates the fundamental principle of using a weighting function to modify the influence of neighbors.

Another approach is to use a Gaussian kernel, which involves an additional parameter for controlling its spread.

```python
import numpy as np
from scipy.spatial import distance
from collections import defaultdict
from math import exp

def gaussian_kernel_weighting(dist, bandwidth):
        """Computes the weight based on the gaussian kernel
        Args:
            dist(float): distance of a neighbour
            bandwidth(float): gaussian kernel width
        Returns:
            float: weight using gaussian kernel
        """
        return exp(- (dist**2)/ (2 * bandwidth**2))


def weighted_knn_classify_gaussian(train_data, train_labels, test_point, k, weighting_function, bandwidth):
    """
    Classifies a test point using weighted k-NN with gaussian weighting.

    Args:
        train_data (np.ndarray): Training data points (n_samples, n_features)
        train_labels (np.ndarray): Labels for the training data (n_samples,)
        test_point (np.ndarray): Test data point (n_features,)
        k (int): Number of neighbors to consider.
        weighting_function (function): Function to compute the weight for distance.
         bandwidth (float): controls the spread of the gaussian curve

    Returns:
        predicted_label (int): The predicted class label for the test point.
    """

    distances = [distance.euclidean(test_point, train_point) for train_point in train_data]
    #Sort the distances and extract k nearest distances and corresponding indices
    k_nearest_indices = np.argsort(distances)[:k]

    k_nearest_distances = np.array(distances)[k_nearest_indices]
    k_nearest_labels = train_labels[k_nearest_indices]

    weighted_votes = defaultdict(float)
    for i, dist in enumerate(k_nearest_distances):
         weight = weighting_function(dist, bandwidth)
         weighted_votes[k_nearest_labels[i]] += weight
    
    # Return the label with max weighted votes
    return max(weighted_votes, key=weighted_votes.get)


# Example usage
if __name__ == '__main__':
    # Generate some dummy training data
    train_data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9,11]])
    train_labels = np.array([0, 0, 1, 1, 0, 1])
    test_point = np.array([2, 2])
    k = 3
    bandwidth = 1.0
    predicted_label = weighted_knn_classify_gaussian(train_data, train_labels, test_point, k, gaussian_kernel_weighting, bandwidth)
    print(f"Test point classified as {predicted_label} using gaussian weighting")

```

This extended example implements the Gaussian kernel. The bandwidth parameter controls the kernel width, affecting how rapidly the weights decrease with distance. A small bandwidth means that only very close neighbors have significant influence. Conversely, a larger bandwidth gives a wider range of neighbors more weight. This requires careful tuning through cross-validation to achieve optimal results.

Finally, I can show an example that demonstrates implementation of exponential decay weighting.

```python
import numpy as np
from scipy.spatial import distance
from collections import defaultdict
from math import exp

def exponential_decay_weighting(dist, bandwidth):
    """Computes the weight based on the exponential decay
        Args:
            dist(float): distance of a neighbour
            bandwidth(float): controls rate of decay
        Returns:
            float: weight
        """
    return exp(- dist/ bandwidth)


def weighted_knn_classify_expdecay(train_data, train_labels, test_point, k, weighting_function, bandwidth):
    """
    Classifies a test point using weighted k-NN with exponential decay.

    Args:
        train_data (np.ndarray): Training data points (n_samples, n_features)
        train_labels (np.ndarray): Labels for the training data (n_samples,)
        test_point (np.ndarray): Test data point (n_features,)
        k (int): Number of neighbors to consider.
        weighting_function (function): Function to compute the weight for distance.
         bandwidth (float): controls the spread of the decay curve

    Returns:
        predicted_label (int): The predicted class label for the test point.
    """

    distances = [distance.euclidean(test_point, train_point) for train_point in train_data]
    #Sort the distances and extract k nearest distances and corresponding indices
    k_nearest_indices = np.argsort(distances)[:k]

    k_nearest_distances = np.array(distances)[k_nearest_indices]
    k_nearest_labels = train_labels[k_nearest_indices]

    weighted_votes = defaultdict(float)
    for i, dist in enumerate(k_nearest_distances):
         weight = weighting_function(dist, bandwidth)
         weighted_votes[k_nearest_labels[i]] += weight
    
    # Return the label with max weighted votes
    return max(weighted_votes, key=weighted_votes.get)


# Example usage
if __name__ == '__main__':
    # Generate some dummy training data
    train_data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9,11]])
    train_labels = np.array([0, 0, 1, 1, 0, 1])
    test_point = np.array([2, 2])
    k = 3
    bandwidth = 1.0
    predicted_label = weighted_knn_classify_expdecay(train_data, train_labels, test_point, k, exponential_decay_weighting, bandwidth)
    print(f"Test point classified as {predicted_label} using exponential decay weighting")
```
This implementation showcases exponential decay with another tunable bandwidth parameter. Smaller bandwidth values causes faster decay.

In practice, choosing the correct weighting function and its parameters requires experimentation. Consider books on machine learning and specifically those focused on non-parametric methods. Additionally, research papers discussing distance metrics and their impact on k-NN performance provide useful guidelines. For example, exploration of different kernels is a common topic in non-parametric statistics. I've found careful cross-validation of these techniques to be essential. Choosing the best combination of a weighting function, parameter, and k, is dependent on the specifics of each unique dataset.
