---
title: "Which axes are appropriate for using the dot product to evaluate a listwise learning-to-rank model's output?"
date: "2025-01-30"
id: "which-axes-are-appropriate-for-using-the-dot"
---
The dot product, when applied to learning-to-rank models, primarily evaluates the *alignment* between predicted relevance scores and the actual relevance ordering of documents within a query. This alignment is crucial for assessing a model's effectiveness in placing highly relevant documents higher in the ranked list. Specifically, I've found the dot product most meaningful when used to assess the *document axis*, meaning I focus on dot products that operate on vectors associated with each individual document.

Here's why that focus is essential. Learning-to-rank models, whether point-wise, pair-wise, or list-wise, aim to produce a ranking over a set of documents associated with a single query. The final output, therefore, is a list of relevance scores. These scores can be used for sorting and ordering to deliver the final ranked output. When we think about using the dot product to evaluate how well a model is doing, the operation has to correlate these predicted scores against something. There are two basic vectors I have found myself considering. The first are the scores themselves. The second are a vector representing ground truth relevance. The key is that we *must* have a vector of ground-truth relevancy. Without that, the dot product doesn’t have a useful reference. For listwise learning, this ground truth can be represented as binary or graded relevancy levels. I treat these as vectors associated with each document when performing the dot product.

The dot product, mathematically, is the sum of the products of corresponding elements in two vectors. Let's say we have a vector *p* representing the model’s predicted scores and *r* representing the true relevance scores of a document. The dot product *p* · *r* becomes a measure of how much the directionality of the predicted and true relevance vectors align. If predicted and actual relevance scores are highly correlated, meaning high relevance scores are assigned to documents with a high actual relevance, then the dot product tends towards a larger positive value. Conversely, if the model is poor, the dot product can be near zero, or even negative if there's an inverse correlation. It is the nature of ranking, not direct prediction of an absolute relevance number, that makes the use of the dot product so suitable for measuring performance when dealing with an output that is a list of scores to be ranked.

When we consider other possible vectors, like query-related features, for example, the dot product is usually not applicable. The dot product needs vectors to be associated with the same entity - we can compare scores for an entity or we can compare predictions and ground truths for an entity. Query-related features are attributes of the query, not of the ranked entity (i.e., the document). Similarly, vectors encoding user behavior, such as clicks, would not be suitable for dot product analysis directly against the relevance scores; they require a different kind of evaluation metric, like precision, recall, or NDCG (Normalized Discounted Cumulative Gain). Those other evaluation metrics are generally better equipped to handle complex, multi-faceted, or non-linear correlations and relationships between model output and the behavior of an end-user.

Here are three illustrative code examples, all in a Python-like syntax for clarity:

**Example 1: Binary Relevance**

```python
import numpy as np

def calculate_dot_product(predicted_scores, true_relevance):
    """Calculates the dot product between predicted scores and true binary relevance scores."""
    predicted_scores = np.array(predicted_scores)
    true_relevance = np.array(true_relevance)
    return np.dot(predicted_scores, true_relevance)

# Example usage
predicted_scores_1 = [0.9, 0.7, 0.3, 0.1]
true_relevance_1 = [1, 1, 0, 0] # binary relevance: relevant or not relevant
dot_product_1 = calculate_dot_product(predicted_scores_1, true_relevance_1)
print(f"Binary Relevance Dot Product: {dot_product_1}") # Output will be 1.6

predicted_scores_2 = [0.1, 0.3, 0.7, 0.9] # poorly ranked output
true_relevance_2 = [1, 1, 0, 0]
dot_product_2 = calculate_dot_product(predicted_scores_2, true_relevance_2)
print(f"Binary Relevance Dot Product (Bad Model): {dot_product_2}") # Output will be 0.4
```

In this example, we have a list of predicted scores, and a binary vector representing true relevance. A document is marked '1' if it's relevant, and '0' if not. I use numpy to efficiently compute the dot product. When the model scores the relevant documents higher than non-relevant, the dot product is larger. It has the property of giving more reward to models that get relevant documents into the top spots. The dot product of the poor ranking is lower than the good ranking as expected.

**Example 2: Graded Relevance**

```python
import numpy as np

def calculate_dot_product(predicted_scores, true_relevance):
    """Calculates the dot product between predicted scores and true graded relevance scores."""
    predicted_scores = np.array(predicted_scores)
    true_relevance = np.array(true_relevance)
    return np.dot(predicted_scores, true_relevance)

# Example usage
predicted_scores_3 = [0.9, 0.7, 0.3, 0.1]
true_relevance_3 = [3, 2, 1, 0] # graded relevance (e.g., 'perfect', 'good', 'fair', 'not relevant')
dot_product_3 = calculate_dot_product(predicted_scores_3, true_relevance_3)
print(f"Graded Relevance Dot Product: {dot_product_3}") # Output will be 4.4

predicted_scores_4 = [0.1, 0.3, 0.7, 0.9] # poorly ranked output
true_relevance_4 = [3, 2, 1, 0]
dot_product_4 = calculate_dot_product(predicted_scores_4, true_relevance_4)
print(f"Graded Relevance Dot Product (Bad Model): {dot_product_4}") # Output will be 1.6
```

Here, the true relevance is not binary but graded, allowing for different levels of relevance. This example demonstrates that the dot product can effectively capture the nuances of graded relevance. This is often more valuable than just binary relevance, because it is a less coarse model of truth. The 'good' model produces a significantly higher score than the bad model, as expected.

**Example 3:  Inappropriate Dot Product Usage**

```python
import numpy as np

def calculate_dot_product(predicted_scores, query_features):
    """Demonstrates INAPPROPRIATE dot product calculation between predicted scores and query features."""
    predicted_scores = np.array(predicted_scores)
    query_features = np.array(query_features)
    return np.dot(predicted_scores, query_features)

# Example Usage: Do NOT do this.
predicted_scores_5 = [0.9, 0.7, 0.3, 0.1]
query_features_5 = [1.0, 0.5, 0.2, 0.8]  # example query feature vector - this is NOT related to any specific document.
dot_product_5 = calculate_dot_product(predicted_scores_5, query_features_5)
print(f"Inappropriate Dot Product: {dot_product_5}") # The output has no interpretable meaning.
```

This last example is deliberately flawed. I show that a dot product of predicted scores with some random vector is not appropriate. This is because the vector of query features has no direct correspondence to the entity that the scores are being applied to. This calculation has no meaningful interpretation. The result is a numerical value without any correlation to the performance of the listwise learning model.

In conclusion, the dot product is most useful when applied between predicted scores and true relevance vectors for *individual documents* within the context of evaluating learning-to-rank model output. This is how I've consistently seen it leveraged across a variety of retrieval systems in practice. It provides a simple, computationally efficient metric to assess the model's ability to align predicted scores with actual relevance ordering. Other metrics, like NDCG, or precision and recall are often needed for evaluation but the dot product is a solid starting point to get an early indicator. It gives an early signal of how good the model is for correctly ordering documents by relevance, at the document axis.

For further learning about evaluation methodologies, I’d recommend exploring literature on information retrieval metrics. Works focusing on list-wise learning and ranking evaluation specifically will be particularly insightful. Resources specializing in information retrieval and ranking evaluation will be useful in your continued learning journey.
