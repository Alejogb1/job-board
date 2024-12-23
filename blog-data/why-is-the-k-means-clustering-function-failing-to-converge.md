---
title: "Why is the k-means clustering function failing to converge?"
date: "2024-12-23"
id: "why-is-the-k-means-clustering-function-failing-to-converge"
---

Alright, let’s dive into this common headache: k-means failing to converge. I've seen this pop up more times than I care to recall, especially back in my days working with large geospatial datasets. Trust me, a non-converging k-means algorithm is a real patience tester. It's rarely a single issue, more like a constellation of potential problems interacting. Let’s unpack some of the primary culprits.

Fundamentally, k-means operates by iteratively assigning data points to clusters, recalculating cluster centroids, and repeating until no point changes clusters. Convergence, in this context, means that the cluster assignments and centroids stabilize; no further movement or adjustments occur. When that *doesn't* happen, it points to one or more issues preventing that stability.

One key issue often stems from poorly initialized centroids. The k-means algorithm relies heavily on its starting points. If these initial centroids are too close together or, worse, fall within sparsely populated regions, the algorithm can struggle to find distinct clusters, essentially bouncing around in a suboptimal state. This can manifest as continuous reassignments and centroid updates without any meaningful convergence. Imagine plotting all data points on a 2D plane; if your initial centroid points are clustered tightly in one corner, you’re starting from a skewed position.

Another frequent culprit is local minima. The objective function of k-means, which aims to minimize the within-cluster sum of squares, isn't globally convex. This means that the iterative process can become trapped in a local minimum, where moving the centroids any further only *increases* the sum of squares, effectively halting the process prematurely without achieving optimal clustering. You'll often see oscillations where the algorithm gets stuck, shifting assignments and recalculating centroids but never truly settling.

Then we have the inherent limitations of the k-means algorithm itself. It assumes clusters are spherical and roughly equal in size, which is rarely the case in real-world data. When clusters are highly irregular or vary significantly in density, k-means might struggle to find the correct boundaries, causing continuous fluctuation in assignments as it tries to force the data into an inappropriate shape. Consider, for instance, clusters that resemble crescent shapes; k-means will likely split them along an arbitrary line and then continuously readjust, never finding a stable solution.

Moreover, issues within the data itself are very common. Outliers, for example, can severely distort the centroids, especially in the initial stages. If a single data point is significantly far removed from the rest of the dataset, the algorithm will attempt to reconcile this outlier with the closest cluster, leading to distorted centroid locations and hindered convergence. Similarly, the existence of many data points lying equidistant to multiple centroids leads to ‘flipping’ between clusters, further perpetuating the non-convergence state.

Let’s look at some code examples to illustrate these problems and solutions. I'll be using python with the scikit-learn library, which is quite common for k-means implementations.

**Example 1: Poor Initialization**

This first example demonstrates how bad centroid initialization can lead to non-convergence within a specific number of iterations. It's not technically 'non-convergence' because `sklearn` will always terminate after a set amount of iterations, but rather a sub-optimal convergence.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Manually initialize centroids to a clustered position
initial_centroids = np.array([[0.1, 0.1], [0.2, 0.2], [0.3,0.3]])

# Run K-means with manual initialization (bad initialization)
kmeans_bad = KMeans(n_clusters=3, init=initial_centroids, n_init=1, max_iter=10)
kmeans_bad.fit(X)

# Run K-means with "k-means++" initialization
kmeans_good = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=10) # increased n_init
kmeans_good.fit(X)

# Plotting (for visualization)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_bad.labels_, cmap='viridis')
plt.scatter(kmeans_bad.cluster_centers_[:, 0], kmeans_bad.cluster_centers_[:, 1], marker='x', s=200, c='red')
plt.title('Bad Initialization')
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_good.labels_, cmap='viridis')
plt.scatter(kmeans_good.cluster_centers_[:, 0], kmeans_good.cluster_centers_[:, 1], marker='x', s=200, c='red')
plt.title('k-means++ Initialization')
plt.show()

```

Here, `n_init` determines how many times the k-means algorithm is run with different centroid seeds, and the best result (defined by inertia – the sum of squared distances of samples to their closest cluster center) is selected as the final clustering solution. By using `n_init=10`, we increase the probability of finding a more optimal solution. `k-means++` is a specific initialization technique that intelligently selects initial centroids, improving algorithm stability and final performance.

**Example 2: Local Minima Traps**

The second example, while less pronounced in simpler datasets, shows how local minima can lead to different results depending on initialization, though you won't see an explicit 'non-convergence' issue with `sklearn`.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate non-convex data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# Run K-means with two different random states (different init)
kmeans1 = KMeans(n_clusters=2, init='random', n_init=10, max_iter=10, random_state=1)
kmeans1.fit(X)

kmeans2 = KMeans(n_clusters=2, init='random', n_init=10, max_iter=10, random_state=2)
kmeans2.fit(X)

# Plotting (for visualization)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans1.labels_, cmap='viridis')
plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], marker='x', s=200, c='red')
plt.title('First Random Init')
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans2.labels_, cmap='viridis')
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], marker='x', s=200, c='red')
plt.title('Second Random Init')
plt.show()

```

Here, the `make_moons` dataset creates non-convex clusters, which are problematic for k-means. Each random initialization (controlled by `random_state`) can converge to a slightly different local optimum. Running `n_init` multiple times helps to mitigate this to some degree.

**Example 3: Preprocessing for Outliers**

The third example addresses the impact of outliers using a robust scaling approach as an intermediary step.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Generate sample data with an outlier
X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.60, random_state=0)
X = np.vstack((X, [10, 10])) # add an outlier


# Run K-means without scaling
kmeans_no_scaling = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=10)
kmeans_no_scaling.fit(X)


# Scale data using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Run k-means with scaled data
kmeans_scaled = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=10)
kmeans_scaled.fit(X_scaled)

# Plotting (for visualization)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_no_scaling.labels_, cmap='viridis')
plt.scatter(kmeans_no_scaling.cluster_centers_[:, 0], kmeans_no_scaling.cluster_centers_[:, 1], marker='x', s=200, c='red')
plt.title('Without RobustScaler')

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_scaled.labels_, cmap='viridis')
plt.scatter(kmeans_scaled.cluster_centers_[:, 0], kmeans_scaled.cluster_centers_[:, 1], marker='x', s=200, c='red')
plt.title('With RobustScaler')
plt.show()
```

Robust scaling is useful here because it’s less sensitive to outliers compared to standard scaling techniques. The outlier in the data biases the centroid computation when the data is unscaled, where as a more optimal result is achieved when data is scaled beforehand.

In my experience, the best way to approach non-converging k-means is systematic troubleshooting. Firstly, increase `n_init` to make sure you're getting a solution as close to a global minima as possible. Then, examine your data for outliers and consider using data preprocessing methods like robust scaling. If non-convex clusters are present consider using DBSCAN as an alternative. Finally, try different initialization methods if `k-means++` isn’t resolving the issue.

For further reading, I would suggest reviewing the original k-means papers by Stuart Lloyd (you can find "Least Squares Quantization in PCM" from 1957, which essentially describes k-means algorithm) and James MacQueen ("Some Methods for classification and Analysis of Multivariate Observations" from 1967). Additionally, "Pattern Recognition and Machine Learning" by Christopher Bishop has a comprehensive overview of clustering algorithms and related mathematical foundations. Another excellent resource is “The Elements of Statistical Learning” by Hastie, Tibshirani, and Friedman; it provides a theoretical and practical approach to understanding k-means limitations and potential remedies. Understanding these texts will provide a much deeper understanding of the fundamentals and limitations of k-means than any single example I could ever provide.
