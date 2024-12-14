---
title: "How to use KNN to find outlier in a grouped pandas dataframe?"
date: "2024-12-14"
id: "how-to-use-knn-to-find-outlier-in-a-grouped-pandas-dataframe"
---

alright, so you're looking at using k-nearest neighbors (knn) to find outliers in a pandas dataframe, where you've got some sort of grouping going on, yeah? i've been down this rabbit hole a few times, and it's a pretty common scenario when dealing with complex datasets. let me share what i've learned, and some code snippets that'll hopefully get you on the right track.

the core idea here is that you're not just looking for outliers in the entire dataset, but you want to find them *within* each group. think about it like this: imagine you have data on people's heights, but split by country. a height that's unusual in a group from south-east asia might be perfectly normal in northern europe. that's why we need to apply knn for outlier detection within each group separately.

my first time tackling something like this, i was dealing with telemetry data from a bunch of sensors on industrial machines, split by machine type. i tried global outlier detection at first, and it was a total mess. it was flagging normal readings on one machine type as outliers just because it had a very different operating range than others. it was a facepalm moment to put it mildly. i wasted a whole day trying to understand why my perfectly reasonable models were going mad. lesson learned hard.

so, how does it work in practice with pandas and knn? the trick is to use the `groupby` operation to process each group individually. for each group, we use knn to compute a "distance score" for each point. the points that have very high distance scores, those are your outliers.

here's a simplified example using scikit-learn and pandas:

```python
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

def find_outliers_knn(df, group_column, feature_columns, n_neighbors=5, threshold_factor=2.0):
    """
    find outliers in a pandas dataframe using knn within groups.

    args:
    df (pd.dataframe): input dataframe.
    group_column (str): column name to group by.
    feature_columns (list): list of column names to use as features.
    n_neighbors (int): number of neighbors for knn.
    threshold_factor (float): factor to multiply std by to set outlier threshold.

    returns:
    pd.dataframe: dataframe with an 'is_outlier' column.
    """
    def _calculate_outlier_score(group):
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        features = group[feature_columns].values
        knn.fit(features)
        distances, _ = knn.kneighbors(features)
        avg_distances = distances.mean(axis=1)
        return avg_distances

    df['outlier_score'] = df.groupby(group_column).apply(_calculate_outlier_score).values
    thresholds = df.groupby(group_column)['outlier_score'].transform(lambda x: x.mean() + threshold_factor * x.std())
    df['is_outlier'] = df['outlier_score'] > thresholds
    return df
# Example usage
data = {
    'machine_type': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c','a','b','c'],
    'temperature': [25, 26, 27, 30, 31, 50, 20, 22, 10, 29, 28, 21],
    'pressure': [100, 102, 101, 120, 122, 105, 95, 98, 100, 103, 119, 96]
}
df = pd.dataframe(data)
feature_columns = ['temperature', 'pressure']
result_df = find_outliers_knn(df, 'machine_type', feature_columns)
print(result_df)
```

in this snippet, `find_outliers_knn` function does the work. it groups the dataframe by the specified column, then calculates the average knn distance for each point. i am using average instead of sum because it avoids large values from having many neighbours. then it flags anything as an outlier that is further than some standard deviations away from the mean. the `threshold_factor` allows adjusting the sensitivity to what is an outlier. this factor might need to be tuned for your data and that was one of the most tedious parts of my previous job.

you might be asking yourself 'why not just take the max value instead of using the threshold with the std?' well, using `std` helps us to catch those outliers that are consistently at a further distance instead of catching a few random cases. this method provides a more data based approach.

now, a few notes on this code. i'm using `sklearn.neighbors.nearestneighbors` because it's a fairly standard library for knn tasks. also, i'm using euclidean distance as a default metric. if your data has a strange distribution, this might not be a good metric. you might need to do some research and pick the best distance metric for your case. or if you have categoric values that you want to use as well, you have to encode them first or use metrics that handle those like hamming distance. it is important to note that this simple example lacks preprocessing steps but should work for most cases.

let's delve deeper. you might encounter situations where the data scales for different features are dramatically different. this could cause features with larger values to dominate the distance calculation, leading to incorrect results. we can solve it by scaling the data with `standardscaler` or `minmaxscaler` from scikit-learn, before passing them into the knn model. i cannot stress enough how many hours i lost debugging because of this very simple mistake. a single digit value for temperature can be comparable to values in the thousands for pressure, making the distance calculations very wrong.

here's how you would modify the function to incorporate scaling. the code has changed a lot from the previous one so make sure you compare them side by side to find the changes.

```python
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

def find_outliers_knn_scaled(df, group_column, feature_columns, n_neighbors=5, threshold_factor=2.0):
    """
    find outliers in a pandas dataframe using knn within groups, with scaling.

    args:
    df (pd.dataframe): input dataframe.
    group_column (str): column name to group by.
    feature_columns (list): list of column names to use as features.
    n_neighbors (int): number of neighbors for knn.
    threshold_factor (float): factor to multiply std by to set outlier threshold.

    returns:
    pd.dataframe: dataframe with an 'is_outlier' column.
    """
    def _calculate_outlier_score(group):
        features = group[feature_columns].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(scaled_features)
        distances, _ = knn.kneighbors(scaled_features)
        avg_distances = distances.mean(axis=1)
        return avg_distances

    df['outlier_score'] = df.groupby(group_column).apply(_calculate_outlier_score).values
    thresholds = df.groupby(group_column)['outlier_score'].transform(lambda x: x.mean() + threshold_factor * x.std())
    df['is_outlier'] = df['outlier_score'] > thresholds
    return df

# Example usage
data = {
    'machine_type': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c','a','b','c'],
    'temperature': [25, 26, 27, 30, 31, 50, 20, 22, 10, 29, 28, 21],
    'pressure': [100, 102, 101, 120, 122, 105, 95, 98, 100, 103, 119, 96]
}
df = pd.dataframe(data)
feature_columns = ['temperature', 'pressure']
result_df = find_outliers_knn_scaled(df, 'machine_type', feature_columns)
print(result_df)
```

here, i've added `standardscaler` to scale the features before feeding them into the knn algorithm. the `standardscaler` transforms your features into a standard normal distribution with zero mean and unit variance. you might want to change this to `minmaxscaler` if the distribution of the data is very skewed or use other scalers available at scikit-learn. it's critical to understand the properties of the data to choose the right scaling technique.

another improvement you might want to consider is how you handle the number of neighbors. a value of `n_neighbors` that's too small can make your outlier detection very sensitive to local variations, which can cause noise to be marked as an outlier. if the value of `n_neighbors` is too large it might miss the real outliers. you might want to experiment with this parameter to find the sweet spot for your specific dataset.

for complex or very large datasets, you might need to look at other ways to make the algorithm more efficient. for instance, if you have a very very very large number of samples, the brute-force approach of `sklearn.neighbors.nearestneighbors` will become very slow. you may then want to explore algorithms like ball tree or k-d tree, which offers significant performance improvements, especially for high-dimensional data. or perhaps you should consider approximate knn using libraries like annoy or nmslib.

finally, consider if knn is actually the best tool for the job. although knn is a powerful and versatile algorithm, you might find other outlier detection methods like isolation forest or one-class svm more effective. it all depends on your data distribution and what kind of outliers you are searching for. they each have different characteristics and are more suitable for different scenarios. if your outliers are very different you might even want to use a simple statistical approach like z-score or iqr (interquartile range). there is no silver bullet when it comes to outlier detection.

and now a joke to ease the technical burden: why did the programmer quit his job? because he didn't get arrays.

as for resources, i'd recommend "pattern recognition and machine learning" by christopher bishop for a deep dive into the theoretical aspects of knn and machine learning. it covers a lot of fundamental concepts. also, for a more practical approach, "hands-on machine learning with scikit-learn, keras & tensorflow" by aurelien geron is a great book for getting started with machine learning algorithms. both are great and a very good place to start.

so, to sum things up, using knn for outlier detection within grouped data can be powerful, but it's crucial to understand the steps involved, from the correct data preparation using scaling, and choosing the right amount of neighbors, and the limitations of the algorithm. always validate your assumptions and consider alternatives when things go wrong, and when they do go wrong, consider re-reading the books i recommended. good luck with your outlier hunting!
