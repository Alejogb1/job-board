---
title: "How many observations are different between cluster models?"
date: "2024-12-23"
id: "how-many-observations-are-different-between-cluster-models"
---

Okay, let's tackle this. I remember back in my days at *Acme Analytics*, we faced a very similar puzzle. We were trying to refine our customer segmentation, and a colleague had generated multiple clustering solutions, each using slightly different feature sets or algorithms. The critical question then, as it is now, was: how do we quantify the *dissimilarity* between these different clusterings? Saying "they look a bit different" simply wasn't going to cut it. We needed concrete metrics to determine if our changes were moving us toward a more meaningful segmentation or simply rearranging noise.

The core challenge with comparing cluster models is that unlike classification where you have ground truth labels to measure against, cluster assignments are inherently ambiguous. Different algorithms, or even the same algorithm with different initialization points, can produce perfectly valid but wholly different cluster structures. The direct comparison of cluster labels is often meaningless. What we need are ways to look at how the *underlying data points* are being grouped, rather than focusing on the arbitrary cluster IDs.

One of the first, and simplest, approaches we explored was based on *pairwise comparisons*. Imagine we have two clustering results, *C1* and *C2*, both applied to the same dataset. We can go through every pair of data points and check, in *C1*, are these two points in the same cluster, or in different clusters? Then we do the same check in *C2*. If the two points are grouped similarly in both clusterings, great. But if they are grouped differently, that's an indication of disagreement between the two models. We can then sum up the disagreements across all point-pairs to get a sense of how different these clusterings are.

This approach is formalized by a metric called the *Rand Index* and a related metric called the *Adjusted Rand Index* (ARI). While the Rand Index can be useful, its unadjusted nature leads to issues, specifically where random agreements can push the score higher. The ARI, a correction for this, gives a better metric for comparison, with a value of 1 meaning identical groupings and values approaching 0 indicating essentially random agreement between the groupings. Negative values are possible and are often interpreted as a model being no better than a random assignment.

Here's a basic python implementation of ARI using `scikit-learn`:

```python
from sklearn.metrics import adjusted_rand_score
import numpy as np

# Example cluster assignments for two models
cluster1_assignments = np.array([0, 0, 1, 1, 2, 2])
cluster2_assignments = np.array([0, 1, 1, 0, 2, 2])


ari_score = adjusted_rand_score(cluster1_assignments, cluster2_assignments)
print(f"Adjusted Rand Index: {ari_score}")

#Example with more disagreement:
cluster3_assignments = np.array([0, 1, 2, 0, 1, 2])
ari_score_3 = adjusted_rand_score(cluster1_assignments, cluster3_assignments)
print(f"Adjusted Rand Index between C1 and C3: {ari_score_3}")
```

In that snippet, you'll see how different the resulting scores are, reflecting how different the cluster arrangements are. A high score implies very high similarity while a score close to zero reflects that most assignments are not the same, even by coincidence. In our initial experiments, this quickly became our go-to method for comparing different runs of the same algorithm, as well as evaluating the impact of using, for instance, k-means vs. a hierarchical clustering method. However, the ARI treats all disagreements equally, which can be a problem, as it doesn't take into account cluster sizes or distances between points.

Another perspective we took, particularly when we had a more *hierarchical* view of our data, involved measuring the difference in cluster membership using an *overlap coefficient*, often expressed as the Jaccard Index at the cluster level. Rather than pairwise comparisons of individual data points, we focused on the intersection of the clusters themselves. If two clusters from *C1* and *C2* have many shared members, that's a good sign of similarity. We calculated Jaccard indices by comparing all cluster pairs and either calculated the average, or the maximum, if we were trying to find the best matching cluster for each one.

The Jaccard Index (or Jaccard Similarity Coefficient) gives us the size of the intersection divided by the size of the union between two clusters. This approach allowed us to see if individual clusters from different models were actually representing the same group of data points. It’s particularly useful when the total number of clusters varies, as we’d then have many clusters in one model not be present in another.

Here's an implementation of cluster-wise Jaccard Index calculations, illustrating how it identifies differing cluster compositions:

```python
def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def cluster_jaccard(cluster1_assignments, cluster2_assignments):
  cluster1_sets = {}
  for i, cluster in enumerate(cluster1_assignments):
    if cluster not in cluster1_sets:
      cluster1_sets[cluster] = set()
    cluster1_sets[cluster].add(i)

  cluster2_sets = {}
  for i, cluster in enumerate(cluster2_assignments):
     if cluster not in cluster2_sets:
      cluster2_sets[cluster] = set()
     cluster2_sets[cluster].add(i)

  jaccard_scores = []
  for key1, set1 in cluster1_sets.items():
      for key2, set2 in cluster2_sets.items():
         jaccard = jaccard_index(set1, set2)
         jaccard_scores.append((key1, key2, jaccard))
  return jaccard_scores

# Using our previous example data, we'll apply our jaccard index
jaccard_results = cluster_jaccard(cluster1_assignments, cluster2_assignments)
print("Jaccard Indices between Clusters:")
for cluster1, cluster2, score in jaccard_results:
    print(f"Cluster {cluster1} (C1) vs. Cluster {cluster2} (C2): {score}")
```

This shows the Jaccard Index for each cluster pair, letting us identify specific areas of difference between clusters. When used in conjunction with ARI, this gives a complete picture of how well a set of clusters are aligned.

Finally, a more nuanced method we employed when cluster boundaries were especially important involved the use of *silhouette scores*. Silhouette scores measure how similar a data point is to its own cluster compared to the next nearest cluster. This allowed us to determine whether data points were "well clustered" or whether they straddled cluster boundaries. By averaging these scores across all points within each cluster, we could quantify the overall coherence of each cluster, and then calculate the difference in these scores between our competing models. This works best when clusters are somewhat separated, but it's still a useful metric to monitor the quality of a clustering outcome, although it does not directly compare two different sets of clusters to each other. We can however compare the aggregated silhouette scores for each cluster, or the aggregate silhouette score for each cluster model as a whole.

Here is a simplified example demonstrating how to generate and analyse the average silhouette score for a dataset:

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import make_blobs

#Generating example blobs for use
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

kmeans_c1 = KMeans(n_clusters=3, random_state=42, n_init=10)
c1_labels = kmeans_c1.fit_predict(X)

kmeans_c2 = KMeans(n_clusters=3, random_state=142, n_init=10)
c2_labels = kmeans_c2.fit_predict(X)

# Calculate silhouette scores for each set of clusters
silhouette_avg_c1 = silhouette_score(X, c1_labels)
silhouette_avg_c2 = silhouette_score(X, c2_labels)

print(f"Average silhouette score for cluster model 1: {silhouette_avg_c1}")
print(f"Average silhouette score for cluster model 2: {silhouette_avg_c2}")

```
In this last snippet, we demonstrate how we can compare two clustering solutions of the same data points. While this approach does not tell us how different these clusters are directly, it can give us a useful indication of whether one model creates "tighter" or more separated clusters than the other, and should thus be considered better.

In summary, choosing which approach to take when comparing different cluster models depends a great deal on what you're trying to evaluate. ARI provides a generalized view of the similarity between assignments. The cluster-wise Jaccard allows for the direct comparison of cluster membership. The silhouette score helps us evaluate the quality of clusters without comparing them directly. Having these tools in your arsenal gives you the flexibility to evaluate model performance effectively and efficiently. For further reading on these methods, I would highly recommend exploring the work of Meilă, especially her paper on comparing clusterings, *“Comparing Clusterings - an Information Based Distance”*. Also, *“The Elements of Statistical Learning”* by Hastie, Tibshirani, and Friedman, provides a strong grounding in the statistical framework of many of these metrics, which greatly enhances one's understanding of the underlying principles.
