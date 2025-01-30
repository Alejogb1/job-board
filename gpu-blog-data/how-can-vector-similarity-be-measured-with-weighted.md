---
title: "How can vector similarity be measured with weighted features?"
date: "2025-01-30"
id: "how-can-vector-similarity-be-measured-with-weighted"
---
Vector similarity, at its core, quantifies the likeness between two numerical representations. In scenarios involving feature-rich data, it’s frequently observed that not all features contribute equally to the overall similarity; some are inherently more discriminative than others. Ignoring this disparity can lead to inaccurate similarity assessments. Thus, incorporating feature weights is essential for a more nuanced understanding of vector relationships.

My experience in developing a recommendation engine for academic papers revealed this vividly. We initially used cosine similarity based on unweighted TF-IDF vectors representing the papers. This produced surprisingly poor results; many recommended papers, despite overlapping keywords, were semantically irrelevant. Upon deeper analysis, we realized that certain terms, such as very common research methodologies, were inflating the similarity scores while failing to capture the core research focus. This highlighted the necessity of weighting features based on their informative value.

Weighted vector similarity builds on the established techniques like cosine similarity, Euclidean distance, or dot product, but extends these calculations by assigning a specific weight to each dimension (feature) in the vector space. This weight reflects the significance of that particular feature when measuring the similarity between vectors. For instance, consider two vectors representing text documents; if the feature “quantum physics” is known to be highly informative in our application, we will assign this dimension a large weight. This means documents sharing “quantum physics” will exhibit a larger similarity compared to other, less important, dimensions.

The key concept is applying the weights prior to the similarity computation. Instead of directly using the feature values *x* and *y* of two vectors *A* and *B*, we use the weighted feature values, calculated as *x’ = x*w* and *y’ = y*w*, where *w* is the weight for that specific feature. These weighted values are then used in the similarity calculation method.

Let's explore this with concrete examples using Python and NumPy:

**Example 1: Weighted Cosine Similarity**

This example demonstrates the most common approach: weighted cosine similarity. I frequently used this approach in my prior work on document clustering. The cosine similarity measures the angle between two vectors; weighting influences this angle by scaling the feature values along each dimension.

```python
import numpy as np

def weighted_cosine_similarity(vector_a, vector_b, weights):
    """
    Calculates the weighted cosine similarity between two vectors.

    Args:
        vector_a (np.array): The first vector.
        vector_b (np.array): The second vector.
        weights (np.array): The weights for each feature.

    Returns:
        float: The weighted cosine similarity.
    """
    weighted_a = vector_a * weights
    weighted_b = vector_b * weights

    norm_a = np.linalg.norm(weighted_a)
    norm_b = np.linalg.norm(weighted_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0  # handle cases where one vector has zero magnitude
    return np.dot(weighted_a, weighted_b) / (norm_a * norm_b)

# Example usage:
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])
weights = np.array([0.1, 0.5, 2.0]) # feature 3 has highest weight

similarity = weighted_cosine_similarity(vector_a, vector_b, weights)
print(f"Weighted Cosine Similarity: {similarity:.4f}")
```

In this example, the function first multiplies each feature value by its corresponding weight. Then, the dot product and norms are calculated based on these weighted vectors, and finally, the weighted cosine similarity is computed. The weights vector allows us to bias the similarity result towards the features we believe to be more significant.

**Example 2: Weighted Euclidean Distance**

The Euclidean distance calculates the straight-line distance between two points. When applied with weights, it modifies the contribution of each feature to the overall distance. I found this particularly useful in scenarios where magnitude differences were paramount.

```python
import numpy as np

def weighted_euclidean_distance(vector_a, vector_b, weights):
    """
    Calculates the weighted Euclidean distance between two vectors.

    Args:
        vector_a (np.array): The first vector.
        vector_b (np.array): The second vector.
        weights (np.array): The weights for each feature.

    Returns:
        float: The weighted Euclidean distance.
    """
    weighted_diff = (vector_a - vector_b) * weights
    return np.sqrt(np.sum(weighted_diff**2))


# Example usage:
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])
weights = np.array([0.1, 0.5, 2.0])

distance = weighted_euclidean_distance(vector_a, vector_b, weights)
print(f"Weighted Euclidean Distance: {distance:.4f}")
```

Here, we calculate the difference between feature values, scale the difference by the feature's weight, square the scaled difference, sum the squares, and then take the square root of the sum. This creates a weighted measure of distance that prioritizes features with higher weights.

**Example 3: Generalized Weighted Similarity using a Weighting Function**

This example highlights a more flexible approach. Rather than limiting weights to a single vector, this illustrates the use of a function that generates the weights based on the specific features of both input vectors. This has proven incredibly useful in my work dealing with time-series data, where the importance of specific points can be relative to their position in the series.

```python
import numpy as np

def gaussian_weight(feature_value_a, feature_value_b, mu, sigma):
    """
    Generates a Gaussian weight for a given feature based on the distance between
    the feature values of the two input vectors.

    Args:
        feature_value_a (float): Feature value from the first vector.
        feature_value_b (float): Feature value from the second vector.
        mu (float): The mean of the Gaussian distribution.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        float: The weight generated by the Gaussian distribution.
    """
    distance = abs(feature_value_a - feature_value_b)
    weight = np.exp(-((distance - mu)**2) / (2 * sigma**2))
    return weight

def generalized_weighted_similarity(vector_a, vector_b, weight_func, weight_args):
    """
    Calculates the generalized weighted similarity using a weighting function.

    Args:
        vector_a (np.array): The first vector.
        vector_b (np.array): The second vector.
        weight_func (function): Function used for calculating weights.
        weight_args (dict): Arguments to pass into the weight_func

    Returns:
        float: The generalized weighted similarity.
    """

    weighted_product_sum = 0.0
    norm_sum_a = 0.0
    norm_sum_b = 0.0

    for i in range(len(vector_a)):
      weight = weight_func(vector_a[i], vector_b[i], **weight_args)
      weighted_product_sum += (vector_a[i] * weight) * (vector_b[i] * weight)
      norm_sum_a += (vector_a[i] * weight)**2
      norm_sum_b += (vector_b[i] * weight)**2

    if norm_sum_a == 0 or norm_sum_b == 0:
        return 0.0 # handle zero magnitude cases

    return weighted_product_sum / (np.sqrt(norm_sum_a) * np.sqrt(norm_sum_b))

# Example usage
vector_a = np.array([1, 2, 3, 4, 5])
vector_b = np.array([2, 3, 4, 5, 6])
weight_args = {"mu":0, "sigma": 1}
similarity = generalized_weighted_similarity(vector_a, vector_b, gaussian_weight, weight_args)
print(f"Generalized Weighted Similarity: {similarity:.4f}")
```

This example introduces a Gaussian weighting function. Features where values are closer are given greater weight in this case. The `generalized_weighted_similarity` function then iterates through each feature using this weighting function, calculating similarity as per weighted cosine similarity. This illustrates a custom weighting strategy that allows for more sophisticated handling of feature importance, enabling it to be data-dependent and adaptable. This is significantly more flexible than static weights.

When selecting a similarity metric, consider the nature of your data and the desired emphasis. Cosine similarity is often effective for text data due to its scale-invariance, whereas Euclidean distance might be more appropriate when the magnitude of features holds significant meaning. The use of a weighting function will likely be needed when weighting requirements are highly custom or dependent on interactions between the data.

For further study, several resources provide deeper insights. Texts on information retrieval and recommender systems, such as those exploring the TF-IDF (Term Frequency-Inverse Document Frequency) model and its usage, are invaluable. Furthermore, exploring research literature on embedding techniques, specifically those used in natural language processing, will expose practical methods for representing and manipulating weighted vectors. Examining studies on the mathematics of distance metrics and their implications in different applications will allow for a deeper understanding of the calculations and their applications. Finally, exploring statistical learning materials that cover weighting techniques and dimensionality reduction methods will greatly enhance your understanding of the topic. Utilizing these resources will deepen one's knowledge and enable more effective implementation of weighted vector similarity in diverse contexts.
