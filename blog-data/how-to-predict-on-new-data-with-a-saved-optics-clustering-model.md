---
title: "How to predict on new data with a saved OPTICS clustering model?"
date: "2024-12-15"
id: "how-to-predict-on-new-data-with-a-saved-optics-clustering-model"
---

ah, this brings back memories. i've spent more hours than i care to count staring at clustering algorithms, optics included. seems like you’ve got a trained optics model and now want to unleash it on some fresh, unseen data. i totally get that. it's a common hurdle, and honestly, the scikit-learn documentation, while decent, doesn’t always spell this part out in bold, flashing neon. let me walk you through what i’ve learned after a few painful trial and error sessions over the years, starting with a particularly frustrating project around user behaviour analytics back in my startup days.

so, here’s the deal with optics: it’s not like k-means where you get nice, neat cluster centroids that you can easily measure the distance to. optics builds a reachability plot and from that it extracts the cluster structure based on a parameter epsilon (maximum distance between two data points to be considered in the same neighbourhood) and min_samples (minimum data points in neighbourhood to be considered as a cluster). this is great for finding clusters of varying densities, but it means that directly "predicting" new data isn't straightforward. you can't just feed new points into the algorithm like you would with a classification model.

the core issue is that optics essentially defines clusters based on the *relationships* between your training data points. new data doesn't have those relationships yet, so we need to figure out how it fits into this pre-existing structure. what you actually need to do is assign each new point to the cluster that’s “closest”, or where it has the highest “similarity”, to the points in that cluster. but what we define as “closest” or “similarity” will depend on your needs and context. that’s what took me long time to understand. we need to find an adequate similarity measure.

i think the most reasonable, yet practical way of doing this is:

1. **extract cluster data from the fitted model:** you have to understand, that optics saves the indexes of the training data in each cluster, and we need to use those indexes in our training data to extract the data points for each cluster.
2. **define a distance or similarity function:** you’ll need something that tells us how similar a new point is to an existing cluster. often, simple euclidean distance is more than enough. that was enough for me in my user behaviour analytics project. i used euclidian distance but with a normalization factor in some axis of my multidimensional data, as i had data with different units (e.g. time and website access). i needed to scale some dimensions to properly weight the values.
3. **assign each new point to the most similar cluster:** this means comparing it against every cluster’s center (if you have cluster centers) or against some samples of the cluster and picking the closest or most similar.

here is a code example using python and scikit-learn:

```python
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import pairwise_distances

def predict_optics(optics_model, new_data, training_data):
    """
    predict the cluster for new data based on a fitted optics model

    Args:
        optics_model: fitted optics model
        new_data (np.array): new data to predict
        training_data (np.array): data used in model training

    Returns:
        np.array: the predicted cluster for new data. -1 means outliers

    """
    labels = optics_model.labels_
    clusters = np.unique(labels)
    cluster_predictions = []

    for new_point in new_data:
        best_cluster = -1 # assign -1 (outlier) by default
        min_distance = np.inf
        for cluster_label in clusters:
          if cluster_label == -1:
              continue # skip outlier cluster for distance calculations
          cluster_points = training_data[labels == cluster_label] # get the cluster data from the training data
          distances = pairwise_distances([new_point],cluster_points) # calc the distance from the new point to the cluster points
          average_distance = np.mean(distances) # we could use different distance measures
          if average_distance < min_distance: # pick the cluster with the smallest average distance
              min_distance = average_distance
              best_cluster = cluster_label
        cluster_predictions.append(best_cluster)
    return np.array(cluster_predictions)

if __name__ == '__main__':
    # example usage
    rng = np.random.RandomState(42)
    X = np.concatenate([rng.normal(loc=0, scale=1, size=(100, 2)),
                        rng.normal(loc=5, scale=1, size=(100, 2)),
                        rng.normal(loc=2.5, scale=1, size=(100, 2))]) # create test data

    optics = OPTICS(min_samples=10, xi=.05, min_cluster_size=.05)
    optics.fit(X) # train optics

    new_data = np.array([[2, 2], [6, 6], [-1, -1], [4, 2.5]]) # create new data
    predictions = predict_optics(optics, new_data, X) # predict new data
    print(f"predicted clusters: {predictions}")
```

here, `predict_optics` function takes a fitted `optics` model, your `new_data` and the `training_data` used to train the model. it returns a label for each new data point that corresponds to the closest cluster in the original training data. the function uses euclidean distance and the average distance to the training cluster points to assign a label. we also return `-1` if no cluster is "close enough", as we skip distance computations for outliers, but you could define other criteria to assign outliers.

i want to add another common way of doing this:

*   **calculate reachability of new points:** since the optic model calculates and saves the reachability of data points in training data, a second alternative would be to include the new points in our training data, and recalculate the reachability of the new points in regards to the old ones. we would use our optics model to recalculate reachability of all points and then assign the labels from our original fitted model to the new points. here's how you could do it:

```python
import numpy as np
from sklearn.cluster import OPTICS

def predict_optics_reachability(optics_model, new_data, training_data):
    """
     predict the cluster for new data based on the reachability using a fitted optics model

    Args:
        optics_model: fitted optics model
        new_data (np.array): new data to predict
        training_data (np.array): data used in model training

    Returns:
        np.array: the predicted cluster for new data, -1 means outlier
    """
    combined_data = np.concatenate((training_data, new_data), axis=0) # combine new data with old training data
    new_optics = OPTICS(min_samples=optics_model.min_samples, xi=optics_model.xi, min_cluster_size=optics_model.min_cluster_size)
    new_optics.fit(combined_data) # re-fit optics to combined data
    new_labels = new_optics.labels_ # get the labels
    return new_labels[len(training_data):] # return only the labels of the new data

if __name__ == '__main__':
    # example usage
    rng = np.random.RandomState(42)
    X = np.concatenate([rng.normal(loc=0, scale=1, size=(100, 2)),
                        rng.normal(loc=5, scale=1, size=(100, 2)),
                        rng.normal(loc=2.5, scale=1, size=(100, 2))])  # create test data

    optics = OPTICS(min_samples=10, xi=.05, min_cluster_size=.05)
    optics.fit(X)  # train optics

    new_data = np.array([[2, 2], [6, 6], [-1, -1], [4, 2.5]]) # create new data
    predictions = predict_optics_reachability(optics, new_data, X) # predict new data
    print(f"predicted clusters: {predictions}")
```

this method recalculates the reachability distance for new data and assigns clusters based on it. i am not sure if this is correct though, i remember having a really difficult time understanding the implications of doing this. this could lead to a different clustering structure, so i am not entirely sure if this is valid. please, use it with care. let me know if you find out more about this approach and its validity.

lastly, i will give you another example using the concept of "cluster representation", where we represent the clusters with one or more mean data points, and then we calculate the distance of the new data points to those mean points:

```python
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import pairwise_distances

def predict_optics_representative(optics_model, new_data, training_data, representative_points = 1):
    """
     predict the cluster for new data based on a cluster representation using a fitted optics model

    Args:
        optics_model: fitted optics model
        new_data (np.array): new data to predict
        training_data (np.array): data used in model training
        representative_points (int) : the number of representative points per cluster

    Returns:
        np.array: the predicted cluster for new data, -1 means outlier
    """
    labels = optics_model.labels_
    clusters = np.unique(labels)
    cluster_predictions = []
    representative_clusters = []

    for cluster_label in clusters:
        if cluster_label == -1:
          continue # skip outlier cluster for representation
        cluster_points = training_data[labels == cluster_label]  # get the cluster data from the training data
        # get the representative points of each cluster
        cluster_center = np.mean(cluster_points, axis = 0) # i will use mean to represent the cluster
        representative_clusters.append((cluster_label, [cluster_center])) # store cluster label and representative point

    for new_point in new_data:
        best_cluster = -1
        min_distance = np.inf
        for cluster_label, cluster_repres in representative_clusters:
          distances = pairwise_distances([new_point],cluster_repres) # calc distance from the new point to the cluster representatives
          average_distance = np.mean(distances) # we could use different distance measures
          if average_distance < min_distance:
              min_distance = average_distance
              best_cluster = cluster_label
        cluster_predictions.append(best_cluster)
    return np.array(cluster_predictions)

if __name__ == '__main__':
    # example usage
    rng = np.random.RandomState(42)
    X = np.concatenate([rng.normal(loc=0, scale=1, size=(100, 2)),
                        rng.normal(loc=5, scale=1, size=(100, 2)),
                        rng.normal(loc=2.5, scale=1, size=(100, 2))])  # create test data

    optics = OPTICS(min_samples=10, xi=.05, min_cluster_size=.05)
    optics.fit(X)  # train optics

    new_data = np.array([[2, 2], [6, 6], [-1, -1], [4, 2.5]]) # create new data
    predictions = predict_optics_representative(optics, new_data, X) # predict new data
    print(f"predicted clusters: {predictions}")

```

this last approach is more or less similar to the first approach, but instead of calculating the distance to all points in a given cluster, we represent each cluster by a point, or some representative points. we could use the mean, the median, or some random points of the cluster to represent the cluster, and then, when we have a new data point we calculate the distance to all the cluster representations. the closest representative point will give the cluster id. for this example, i decided to just use the mean, but you could experiment with different approaches.

some final thoughts. you might want to try different distance functions. cosine similarity might be better if you have high dimensional data, for example, if you have text embeddings.  also, the more clusters you have the more "meaningful" this process will be, as this gives you a structure in your data which is what we are trying to do. if you have too many outliers this means that you don't have many meaningful clusters and perhaps you would like to adjust the parameters of the model to allow more "data" to be inside the cluster (smaller `min_samples`, or smaller `min_cluster_size`, smaller `xi`). remember to normalise your data if you have data in different scales.

for further reading, i would suggest checking out the original optics paper. it can be a bit dense, but it provides a deeper understanding of what's happening "under the hood" of the algorithm. a good book i used during my university days is "data mining: concepts and techniques" by jiawei han and micheline kamber. the book explains with detail all the concepts of clustering. it’s a fantastic read and contains a lot of interesting and useful information. also, i found some good articles in the journal "knowledge and information systems", where i also learned about more obscure clustering algorithms.

i hope this helps you. it's a bit of a "hack" around the limitations of the optics prediction workflow, but it gets the job done. oh, and did you hear about the programmer who quit his job? he didn't get arrays, but now he has his freedom to cluster as he wants!
