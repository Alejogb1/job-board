---
title: "How do I calculate class proportions in scikit-learn's K-NN?"
date: "2025-01-30"
id: "how-do-i-calculate-class-proportions-in-scikit-learns"
---
Class proportions within the context of scikit-learn's K-Nearest Neighbors (KNN) algorithm are not directly calculated *by* the algorithm itself during the model fitting process. KNN is a lazy learner; it doesn't explicitly learn a model from the training data. Instead, it memorizes the training instances and performs computations during prediction. The proportions we're interested in emerge *after* a query point is classified, and they represent the distribution of classes among the 'K' nearest neighbors of that query point. This calculation is fundamental in understanding the confidence associated with a particular prediction by the KNN classifier.

I’ve worked on numerous projects involving image classification, and understanding how these proportions work in KNN has been crucial for model debugging and fine-tuning. It's not enough to just know what class gets predicted. We need to know the distribution of labels among the neighbors to assess the robustness of the prediction. For instance, a prediction where 4 out of 5 nearest neighbors belong to one class is typically considered more reliable than one where only 3 out of 5 neighbors belong to the predicted class, with the remaining 2 split across different classes.

Here's a detailed explanation of how this calculation is done. Once the nearest neighbors for a new data point are identified using a distance metric (Euclidean, Manhattan, etc.) and the chosen K value, the labels of those neighbors are examined. The proportion of each class label within that set of K neighbors represents the confidence that the new data point belongs to that class. These proportions are not part of KNN's core logic, but rather a result of the analysis of neighbors' labels surrounding the test data.

I will illustrate this with code, starting with a simple example that you might find yourself implementing frequently in practice. This initial example focuses on getting the basic class proportions.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Example training data
X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train = np.array(['A', 'A', 'B', 'B', 'A', 'C'])

# Label encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize KNN classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train_encoded)

# Example test data point
X_test = np.array([[2, 1]])

# Get the indices of the K nearest neighbors
_, neighbor_indices = knn.kneighbors(X_test)

# Extract the labels of the neighbors
neighbor_labels = y_train_encoded[neighbor_indices[0]]

# Calculate class proportions manually
unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
class_proportions = dict(zip(label_encoder.inverse_transform(unique_labels), counts / len(neighbor_labels)))

print(f"Class Proportions: {class_proportions}")

```

In this first example, I create a simple dataset with three classes, labeled ‘A’, ‘B’, and ‘C’.  I employ scikit-learn’s `LabelEncoder` because KNN classifiers work best when dealing with numerical labels. Then, after fitting a KNN model with `n_neighbors=3`, the `kneighbors` method is used to identify the indices of the nearest neighbors for the test point. Critically, we then take those indices and use them to index back into the training label dataset to pull out the actual class labels for the neighbors. The proportions are manually computed using `np.unique` to count occurrences, and I present the result as a dictionary using the decoded class labels for better readability. This demonstrates how class proportions are calculated from the neighbor's labels after the query is performed.

Often, I have found myself in situations where I need to integrate this functionality more directly with prediction. Below is the next example building on the first one, demonstrating how to make class proportion calculation more concise:

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter


# Example training data
X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train = np.array(['A', 'A', 'B', 'B', 'A', 'C'])

# Label encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize KNN classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train_encoded)

# Example test data point
X_test = np.array([[2, 1]])

# Get predicted class (most common label)
predicted_class = knn.predict(X_test)

# Get the indices of the K nearest neighbors
_, neighbor_indices = knn.kneighbors(X_test)


# Extract the labels of the neighbors
neighbor_labels = y_train_encoded[neighbor_indices[0]]

# Calculate class proportions using Counter for conciseness
neighbor_counts = Counter(neighbor_labels)
total_neighbors = len(neighbor_labels)
class_proportions = {label_encoder.inverse_transform([label])[0]: count / total_neighbors
                     for label, count in neighbor_counts.items()}

print(f"Predicted Class: {label_encoder.inverse_transform(predicted_class)[0]}")
print(f"Class Proportions: {class_proportions}")

```

This example builds on the previous one but implements a more direct way of computing the class proportions using Python’s `Counter` object from the `collections` module. `Counter` makes counting label occurrences more concise and readable.  I also added the predicted class output from `knn.predict`. This example illustrates how one might typically use the calculated class proportions in conjunction with predicted labels.

Finally, my third example introduces using weights, which are often an advanced consideration in KNN and impact the resulting proportions:

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter


# Example training data
X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train = np.array(['A', 'A', 'B', 'B', 'A', 'C'])

# Label encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize KNN classifier with K=3, using 'distance' weights
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train, y_train_encoded)

# Example test data point
X_test = np.array([[2, 1]])

# Get the indices of the K nearest neighbors and distances
distances, neighbor_indices = knn.kneighbors(X_test)

# Extract the labels of the neighbors
neighbor_labels = y_train_encoded[neighbor_indices[0]]


# Calculate class proportions with distance weights
weighted_counts = Counter()
for label, distance in zip(neighbor_labels, distances[0]):
    weighted_counts[label] += 1/ (distance + 1e-8)  # adding a small number to prevent zero division


total_weight = sum(weighted_counts.values())
class_proportions = {label_encoder.inverse_transform([label])[0]: count / total_weight
                     for label, count in weighted_counts.items()}

# Get predicted class (most common label)
predicted_class = knn.predict(X_test)

print(f"Predicted Class: {label_encoder.inverse_transform(predicted_class)[0]}")
print(f"Class Proportions (Weighted): {class_proportions}")

```

Here, I demonstrate calculating class proportions when weights are incorporated. I set `weights='distance'` when defining the classifier, causing closer neighbors to have a higher weight, and thus more influence, on the final prediction. The distances of each neighbor are retrieved, then the weight is the inverse of the distance (with a small constant added to prevent division by zero).  The class proportions are then calculated based on these weighted counts. This is very important when your data has variable distances between points and you wish to give importance to the nearest.

For more information, I recommend consulting the scikit-learn user guide section on Nearest Neighbors. Additionally, introductory books on machine learning often dedicate sections to k-nearest neighbors algorithms, which may provide deeper theoretical insight. General resources on statistics can further your understanding of proportions, distributions, and other related concepts. I have found that experimenting with different datasets and K values and comparing the results with these types of proportion calculations will deepen understanding of KNN's practical operation. This process highlights that these proportions are not calculated during KNN's fitting process, but are instead a result of the prediction step and are vital for understanding prediction confidence.
