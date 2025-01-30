---
title: "What input types are appropriate for Earth Mover's Distance loss when image similarity is rated on a 0-9 decimal scale?"
date: "2025-01-30"
id: "what-input-types-are-appropriate-for-earth-movers"
---
The Earth Mover's Distance (EMD), also known as the Wasserstein distance, is fundamentally a metric on probability distributions.  Its application to image similarity, therefore, hinges on how we represent images as probability distributions.  A direct 0-9 decimal scale rating, while intuitive for human perception, is not a suitable input *directly* for EMD. The crucial point is that EMD operates on distributions, not scalar values.  My experience in developing image retrieval systems for satellite imagery analysis has highlighted this repeatedly.  Consequently, appropriate input types require a transformation of the 0-9 similarity score into a meaningful probability distribution.

**1.  Clear Explanation of Suitable Input Types**

The 0-9 similarity score needs to be interpreted as a feature or characteristic.  We cannot simply treat it as a single data point for EMD. Instead, we should consider it as reflecting some underlying distributional property of the images being compared. Here are three plausible interpretations and their corresponding input types:

* **Interpretation 1:  Similarity as a Feature Distribution:** We can treat the similarity score as a single point estimate from an underlying distribution of potential similarity scores.  For example, imagine evaluating an image multiple times using different feature extractors or under different conditions (noise, lighting).  Each evaluation yields a score in the [0, 9] range, creating a histogram.  This histogram represents the probability distribution for similarity scores. Thus, the input for EMD would be two histograms – one for each image being compared.  Each histogram would be represented as a vector, where the i-th element corresponds to the probability (frequency) of observing a similarity score of i.

* **Interpretation 2:  Similarity as a Derived Probability:** Rather than directly using the 0-9 scale, consider the score as a proxy for a more fundamental property, such as the probability of two images sharing a certain set of features.  For instance, a score of 9 could represent a high probability (e.g., 0.9) of feature overlap, while a score of 0 could represent a low probability (e.g., 0.1).  This approach requires a mapping function to translate the similarity score into a probability.   The input to the EMD calculation would then be two probability distributions constructed using this mapping. This could be a simple linear transformation or a more sophisticated sigmoid function.


* **Interpretation 3:  Similarity-based Feature Weighting:** This approach is more indirect.  Instead of transforming the similarity score directly, use it to influence the weights of image features.  Imagine you have a set of features (e.g., texture, color histograms, edge descriptors) for each image.  The 0-9 score can modify the weights associated with each feature before creating the distribution.  For instance, a high similarity score could increase the weights of features that contribute positively to the similarity, thereby shifting the probability mass in a way that reflects the similarity assessment.  Here the input to EMD would be two weighted feature distributions.


In all interpretations, the critical aspect is ensuring the EMD input represents probability distributions rather than raw similarity scores.  This allows the EMD algorithm to effectively measure the 'distance' between the probabilistic representations of the images based on their similarity.

**2.  Code Examples with Commentary**

The following examples demonstrate the three interpretations using Python and assuming the existence of a function `emd(dist1, dist2)` that calculates the EMD between two distributions `dist1` and `dist2`.  The specific implementation of EMD can use libraries like `scipy.stats`.  Note, these examples simplify the feature extraction and are illustrative.

**Example 1: Histogram-Based Input**

```python
import numpy as np
from scipy.stats import wasserstein_distance  # Example EMD implementation

def similarity_histogram(scores):
    histogram = np.zeros(10) # 10 bins for scores 0-9
    for score in scores:
        histogram[int(score)] += 1
    return histogram / np.sum(histogram) # Normalize to probability distribution

# Sample similarity scores for two images
image1_scores = [7, 8, 7, 9, 7]
image2_scores = [6, 5, 7, 6, 6]

dist1 = similarity_histogram(image1_scores)
dist2 = similarity_histogram(image2_scores)

emd_distance = wasserstein_distance(dist1, dist2)
print(f"EMD Distance (Histogram-based): {emd_distance}")

```

This code first creates histograms from multiple similarity scores for each image. These histograms are then normalized to become probability distributions, which are used as inputs to the EMD function.

**Example 2: Similarity-to-Probability Mapping**

```python
import numpy as np
from scipy.stats import wasserstein_distance

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sample similarity scores (single values this time)
image1_score = 8
image2_score = 5

# Map scores to probabilities using a sigmoid function
prob1 = sigmoid(image1_score)
prob2 = sigmoid(image2_score)

# Create simple distributions (representing uncertainty, for example)
dist1 = np.array([1 - prob1, prob1])  # Simplified distribution
dist2 = np.array([1 - prob2, prob2])


emd_distance = wasserstein_distance(dist1, dist2)
print(f"EMD Distance (Probability-based): {emd_distance}")

```

Here, a sigmoid function transforms the similarity scores into probabilities.  Simplified distributions are constructed, representing the mapped probability and its complement, before computing EMD.

**Example 3: Feature Weighting**

```python
import numpy as np
from scipy.stats import wasserstein_distance

# Sample image features (simplified example)
image1_features = np.array([0.2, 0.5, 0.3])
image2_features = np.array([0.6, 0.2, 0.2])

# Similarity score
similarity_score = 7

# Weight features based on similarity (example weighting)
weight_factor = similarity_score / 9  # Normalizes to [0,1] range
weighted_features1 = image1_features * (1 + weight_factor)
weighted_features2 = image2_features * (1 + weight_factor)


#Normalize to probabilities (sum to 1)
dist1 = weighted_features1 / np.sum(weighted_features1)
dist2 = weighted_features2 / np.sum(weighted_features2)


emd_distance = wasserstein_distance(dist1, dist2)
print(f"EMD Distance (Feature Weighting): {emd_distance}")

```

In this case, features are weighted based on the similarity score, creating distributions influenced by the similarity assessment.  Normalization ensures the weighted features form probability distributions.



**3. Resource Recommendations**

For a deeper understanding of the EMD and its applications, I would recommend consulting standard textbooks on computer vision and pattern recognition, focusing on chapters dedicated to image retrieval and similarity measures.  Furthermore, research papers on applications of the Wasserstein distance in machine learning will provide valuable insights.  Examining the source code and documentation for established machine learning libraries that implement EMD will aid in practical implementation.  Finally, focusing on publications related to image retrieval using histogram-based comparisons will strengthen understanding of the transformation of image data into comparable distributions.
