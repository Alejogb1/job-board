---
title: "silhouette r calculation tutorial?"
date: "2024-12-13"
id: "silhouette-r-calculation-tutorial"
---

Alright so you're asking about silhouette score calculation I get it Been there done that countless times it's a staple in unsupervised learning and clustering evaluation

First off let's break down why we even care about silhouette scores This isn't just some random metric someone threw at us it's actually a pretty intuitive way to gauge how well your data points are clustered Put simply it tells you how similar an object is to its own cluster compared to other clusters A high score means the data point is well-clustered while a low or negative score suggests it might be in the wrong spot or that your clusters are not well-separated

Ok so you want to calculate it Right Here's the lowdown I'll assume you know the basics of clustering algorithms like k-means or hierarchical clustering since you're asking about evaluation but just in case I've included some resources later on

The silhouette score for a single data point 'i' is calculated as follows:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:

*   `a(i)` is the average distance between data point 'i' and all other data points in the same cluster
*   `b(i)` is the minimum average distance between data point 'i' and all data points in each of the other clusters

Let me tell you I had this exact problem about four years ago I was working on a customer segmentation project for an e-commerce platform and things weren't going as planned My initial clustering produced... well let’s just say it wasn't pretty The silhouette scores were hovering around zero which basically means the clusters were overlapping or random

I remember spending a whole weekend debugging my distance calculations and trying different clustering parameters it felt like banging my head against a wall until it dawned on me that my data was not properly scaled and some of the dimensions where more impactful than others because the scale disparity.

The general formula looks pretty simple but the devil is in the details You need to have a proper implementation of calculating `a(i)` and `b(i)` accurately Here's a python example using numpy for distance calculation and scipy for the silhouette metric so you see both ways of calculating it. This should help you avoid the same headaches I had years ago with manual computations:

```python
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

def calculate_silhouette_for_point(data_point_index, data, cluster_labels):
    # Get the cluster label for the data point
    cluster_label = cluster_labels[data_point_index]
    
    # Find the indices of data points in the same cluster
    same_cluster_indices = np.where(cluster_labels == cluster_label)[0]
    
    # Calculate average intra-cluster distance
    if len(same_cluster_indices) > 1:
        same_cluster_distances = pairwise_distances(data[data_point_index].reshape(1, -1), data[same_cluster_indices])
        a_i = np.sum(same_cluster_distances)/(len(same_cluster_indices)-1)
    else:
        a_i = 0  
        
    # Find all other cluster labels
    other_cluster_labels = np.unique(cluster_labels[cluster_labels != cluster_label])
    
    # Check if other clusters exist
    if len(other_cluster_labels) > 0 :
        b_i_values = []
        for other_label in other_cluster_labels:
            other_cluster_indices = np.where(cluster_labels == other_label)[0]
            other_cluster_distances = pairwise_distances(data[data_point_index].reshape(1,-1), data[other_cluster_indices])
            b_i_values.append(np.sum(other_cluster_distances)/len(other_cluster_indices))
        
        b_i = min(b_i_values)
    else:
        b_i = 0
        
    # Handle the case when both a(i) and b(i) are zero
    if a_i == 0 and b_i == 0:
        return 0
    
    # Calculate silhouette score
    s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0
    
    return s_i

def calculate_silhouette_scores(data, cluster_labels):
    n = len(data)
    silhouette_scores = np.array([calculate_silhouette_for_point(i, data, cluster_labels) for i in range(n)])
    return silhouette_scores

if __name__ == '__main__':
    # Example usage:
    # Dummy data (replace with your actual data)
    data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    cluster_labels = np.array([0, 0, 1, 1, 0, 2])  # Example cluster labels

    # Calculate silhouette scores
    silhouette_scores_custom = calculate_silhouette_scores(data, cluster_labels)
    average_silhouette_custom = np.mean(silhouette_scores_custom)
    print("Custom Silhouette Scores for each data point:", silhouette_scores_custom)
    print("Custom Average Silhouette Score:", average_silhouette_custom)
    
    # Calculate silhouette scores using sklearn
    average_silhouette_sklearn = silhouette_score(data, cluster_labels)
    print("Sklearn Average Silhouette Score:", average_silhouette_sklearn)
```

This code shows you both how to calculate silhouette score for each point and the average score using a custom function as well as using a readily available library like sklearn. You may find yourself needing custom implementations if using different distance metrics or if working with extremely big datasets.

Here's a breakdown:

1.  **`calculate_silhouette_for_point` Function:** Calculates the silhouette score for a single data point given the data points cluster labels and the actual data.

2.  **`calculate_silhouette_scores` Function:** calculates the silhouette score for all data points and returns all scores

3.  **Main Block:** Creates some dummy data and cluster labels to showcase the function implementation
   as well as showcase the usage of sklearn silhouette_score to show the final value similarity.

The main result of this code is the print of the average and each point silhouette scores.

A final point before we proceed, make sure you properly compute the distances. Euclidean distance is common but depending on your data other metrics might be more appropriate like Manhattan or Cosine distance. I remember once my distances where completely wrong because I was trying to use euclidean distance with categorical features mixed with numerical features the results were a completely mess. I mean, imagine computing a squared difference between a shoe size and an income of the client.

Another important consideration the range of your silhouette scores the range is [-1, 1]:

*   **Close to 1**: The point is well clustered
*   **Around 0**: The point is near or between clusters
*   **Close to -1**: The point is probably assigned to the wrong cluster

A good clustering model should have high average silhouette score values.

And remember the average silhouette score is just that an average there will be points with both high and low silhouette values so don’t expect all values to be close to one.

Here’s another example but using sklearn clustering algorithm for kmeans and displaying more results:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs

# Generate sample data for clustering
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=1.0, random_state=random_state)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init = 10)
cluster_labels = kmeans.fit_predict(X)


# Calculate average silhouette score
avg_silhouette_score = silhouette_score(X, cluster_labels)

# Calculate silhouette scores for each data point
silhouette_values = silhouette_samples(X, cluster_labels)

# Print results
print(f"Average Silhouette Score: {avg_silhouette_score:.3f}")


# Create silhouette plot
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
    facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
    
ax1.set_title("Silhouette plot")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")
ax1.axvline(x=avg_silhouette_score, color="red", linestyle="--")
ax1.set_yticks([])
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()
```

This example shows the use of the silhouette_samples function to get the silhouette score for each sample point and plots the results

This is helpful for visualizing the result and checking if each cluster is well formed.

Finally here's a more complex example with a bit of explanation on scaling the data in case you are dealing with data with different magnitude ranges which I had a lot of issues with before.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

def calculate_and_plot_silhouette(X, n_clusters, random_state=42):
    """
    Calculates silhouette score and plots results
    """

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init = 10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score
    avg_silhouette_score = silhouette_score(X_scaled, cluster_labels)
    print(f"Average Silhouette Score: {avg_silhouette_score:.3f}")
    
    
    # Create scatter plot of the data with clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolors='black')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f'KMeans Clustering (Silhouette Score: {avg_silhouette_score:.3f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Generate sample data
    n_samples = 300
    n_features = 2
    n_clusters = 4
    random_state = 42
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=1.2, random_state=random_state)
    
    # Add some variation to one of the features for testing the need for scaling
    X[:,1] = X[:,1] * 100

    # Calculate and plot using silhouette metric
    calculate_and_plot_silhouette(X, n_clusters, random_state=random_state)
    
    
    # Testing different number of clusters to see how different silhoutte scores impact the results
    
    calculate_and_plot_silhouette(X, 2, random_state=random_state)
    calculate_and_plot_silhouette(X, 3, random_state=random_state)
    calculate_and_plot_silhouette(X, 5, random_state=random_state)
```

In this example the most important detail is the standardization step. If your features are on different scales this will affect the distance calculation thus affecting the cluster formation and the silhouette score. The code showcases this by adding a magnitude difference in one of the features and applying a scaling technique.

You can further improve your model by iterating over different values of `n_clusters` and evaluating how the silhouette score performs on each iteration. You can see that in the last part of the example where I'm testing different numbers of clusters.

As for resources I recommend “The Elements of Statistical Learning” by Hastie, Tibshirani, and Friedman it’s a classic for machine learning algorithms and the statistical foundations behind them. You can also check “Pattern Recognition and Machine Learning” by Bishop for a more Bayesian perspective. And for practical implementations and use cases “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Géron provides easy to understand use cases and code snippets.

And remember in the end your silhouette score is a tool not the goal you want meaningful clusters that make sense for your specific use case.

I hope this helps! Let me know if there are other specific questions I can help with.
