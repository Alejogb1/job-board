---
title: "How can K-means++ be implemented without using scikit-learn?"
date: "2024-12-23"
id: "how-can-k-means-be-implemented-without-using-scikit-learn"
---

,  I’ve certainly found myself in situations where relying on external libraries, especially for something as fundamental as k-means clustering, just wasn't practical. I recall a project a few years back involving embedded systems and very limited resources where pulling in something as large as scikit-learn was entirely out of the question. We needed a highly optimized, custom solution, and that meant implementing k-means from scratch, or rather, using a library as small as possible. We specifically chose k-means++, which, as you know, significantly improves the initial centroid selection, moving past the sometimes poor results of basic random initialization. So, let's delve into how to do that without the comfortable crutch of scikit-learn.

The core idea behind k-means++ is fairly straightforward, but getting the implementation precisely correct takes a bit of attention to detail. It enhances the standard k-means algorithm by intelligently seeding the initial centroids, thereby mitigating the issue of poor clustering caused by unfavorable starting points. The standard k-means method randomly chooses *k* initial centroids from the dataset, which might result in some initial centroids being clustered together while leaving other areas of the data space unrepresented. K-means++, in contrast, uses a probabilistic method to ensure that these initial centroids are more spread out, potentially leading to faster convergence and more accurate results.

Here’s a breakdown of the core algorithm, starting with the initialization phase:

1.  **First Centroid Selection:** Pick a centroid randomly from the dataset. This is similar to the standard k-means method.

2.  **Subsequent Centroid Selection:** Calculate the squared distance of each data point to its nearest centroid. Then, choose the next centroid randomly, with the probability of a point being chosen being proportional to its squared distance to its nearest centroid. This is where the “++” magic happens. Points further away from the currently selected centroids are more likely to be picked as the next centroids.

3.  **Repeat:** Continue this distance calculation and probabilistic selection process until you have *k* initial centroids.

Once you have your initial centroids, the rest of the algorithm functions as normal k-means:

4.  **Assignment:** Assign each data point to the closest centroid, usually using Euclidean distance.

5.  **Update:** Calculate the new centroids by averaging the data points assigned to each centroid.

6.  **Iteration:** Repeat steps 4 and 5 until the centroid positions or assignments no longer change significantly. The measure of ‘significant change’ can be measured in various ways – often either a small change in centroids’ positions or stabilization in points’ cluster assignments.

Now, let’s move onto some actual code snippets. For demonstration purposes, we’ll use python and numpy because they are generally easy to understand, but this logic should be translatable to other languages.

**Snippet 1: Calculating Euclidean Distance**

First, we need a utility function to compute Euclidean distances, and it’s important to do it efficiently.

```python
import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))
```

This snippet takes two points (represented as lists or arrays) and returns the Euclidean distance between them. Using `numpy` for array operations makes this calculation much more efficient than using loops in standard python.

**Snippet 2: The K-means++ Initialization**

Now, the meat of k-means++ initialization. This snippet demonstrates the procedure to select initial centroids.

```python
import random

def initialize_centroids_kmeans_plusplus(data, k):
    """
    Initializes k centroids using k-means++ algorithm.

    Args:
        data (list of lists or numpy array): The dataset.
        k (int): The number of clusters.

    Returns:
        list of lists: The initialized centroids.
    """
    centroids = []
    first_centroid_index = random.randint(0, len(data) - 1)
    centroids.append(data[first_centroid_index])

    for _ in range(1, k):
        distances = []
        for point in data:
            min_dist = float('inf')
            for centroid in centroids:
                dist = euclidean_distance(point, centroid)
                min_dist = min(min_dist, dist)
            distances.append(min_dist**2) # Squared distances as mentioned in the algo

        total_distance = sum(distances)
        probabilities = [dist / total_distance for dist in distances]

        #Choose centroid using probabilities
        cumulative_probabilities = np.cumsum(probabilities)
        random_number = random.random()

        for index, prob_sum in enumerate(cumulative_probabilities):
            if random_number <= prob_sum:
                centroids.append(data[index])
                break

    return centroids

```

Here, we initialize the first centroid randomly, just as in the standard method. But for subsequent centroids, we calculate distances to the closest centroid already chosen for each data point, sum their squares, and create weighted probabilities, then use those probabilities to select the next centroid in line. This probabilistic method ensures our centroids are reasonably spread out.

**Snippet 3: Implementing k-means with the custom initialization**

Finally, let’s create a very simplified version of the k-means algorithm using our custom initialization function. Note: This version omits several real world considerations like empty clusters or convergence criteria and is purely illustrative.

```python

def kmeans_algorithm(data, k, max_iterations = 100):
    """
    Simplified k-means algorithm using k-means++ initialization.

    Args:
        data (list of lists or numpy array): The dataset.
        k (int): The number of clusters.
        max_iterations (int): Maximum iterations.

    Returns:
        (list of lists, list of ints): Centroids and cluster assignments.
    """
    centroids = initialize_centroids_kmeans_plusplus(data,k)
    assignments = [0]*len(data) #Initialize cluster assignments.
    for _ in range (max_iterations):
        changed = False
        # Assignment Step
        for i, point in enumerate(data):
            min_dist = float('inf')
            new_assignment = -1

            for j, centroid in enumerate(centroids):
                dist = euclidean_distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    new_assignment = j

            if new_assignment != assignments[i]:
                assignments[i]=new_assignment
                changed = True

        #Update step
        new_centroids = []
        for j in range(k):
            cluster_points = [data[i] for i, assignment in enumerate(assignments) if assignment == j]

            if cluster_points: #check for empty clusters
                new_centroids.append(np.mean(cluster_points, axis=0).tolist())
            else: # In rare cases assign to a random point to avoid division by zero/empty clusters
               new_centroids.append(data[random.randint(0, len(data)-1)])

        centroids = new_centroids
        if not changed:
            break

    return centroids, assignments

```
This function now orchestrates everything, using our defined distance function and k-means++ initialization, and iteratively assigns points to the closest centroids and recomputes centroid positions. It returns the final centroids and point assignments.

For further, more rigorous study on the theoretical underpinnings of k-means and k-means++, I’d suggest looking into research papers like "A comparative study of efficient initialization methods for the k-means clustering algorithm" by Fahim et al. as well as delving into chapters in well known texts on machine learning and data mining such as "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman, which offers a mathematically rigorous treatment of clustering methodologies. I also highly recommend "Pattern Recognition and Machine Learning" by Christopher Bishop which goes into detail on the mathematics behind many clustering algorithms. These resources should provide you with a sound theoretical understanding of the algorithms and the trade-offs of their various implementations.

Working through these kinds of implementations firsthand is invaluable. While libraries like scikit-learn are great for quick prototyping, having an in-depth understanding of how these algorithms work under the hood is critical for any serious practitioner. It will also greatly aid in debugging, optimizing and customizing your implementations, especially when working in resource-constrained environments. I hope this explanation with example snippets clarifies the process of implementing k-means++ without relying on external libraries. Good luck and keep exploring.
