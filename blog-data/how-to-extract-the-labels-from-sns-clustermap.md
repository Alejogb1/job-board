---
title: "how to extract the labels from sns clustermap?"
date: "2024-12-13"
id: "how-to-extract-the-labels-from-sns-clustermap"
---

 so you're wrestling with `sns.clustermap` and trying to yank out those pesky labels aren't you Been there done that Probably messed it up a few times too before figuring it out Clustermaps look beautiful yeah but they're like black boxes sometimes when you need to get at the internal workings

I've seen people spend hours staring at those things trying to figure this out so dont feel bad Been coding for like 15 years now mostly Python and data science stuff Ive built entire recommender systems from scratch so I have a good handle on this I promise

The thing is `sns.clustermap` doesnt explicitly give you a method or function directly to grab the labels It doesnt hand them to you on a silver platter no its all about digging in the object model after its rendered

First the most important thing remember that `sns.clustermap` doesnt return the raw data it returns a `ClusterGrid` object This thing is your key to the kingdom Its not a simple matplotib axes object which a lot of people expect

Let's break it down step-by-step because its not that hard once you get it

Usually you'll have something like this at the start of your code

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# lets make some dummy data for example
data = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                    'B': [6, 7, 8, 9, 10],
                    'C': [11, 12, 13, 14, 15],
                    'D': [16, 17, 18, 19, 20],
                    'E': [21, 22, 23, 24, 25]},
                   index=['row1', 'row2', 'row3', 'row4', 'row5'])

clustermap = sns.clustermap(data)
```

So what now Many users try to do `clustermap.axes` or something similar hoping to find the labels there No such luck those axes are just matplotlib axes that have the actual visualization they don't have the row or column names specifically

The secret is in the `dendrogram_row` and `dendrogram_col` attributes of the `ClusterGrid` object These are `Dendrogram` objects which contain the labels in their `reordered_ind` attribute and `labels` attribute

Here is how I would get the row and column labels using `dendrogram_row` and `dendrogram_col` which are attributes in the clustergrid object which `sns.clustermap` returns:

```python
row_labels = clustermap.dendrogram_row.reordered_ind
col_labels = clustermap.dendrogram_col.reordered_ind

# these are indexes not the actual labels
print(f"Reordered row indexes: {row_labels}")
print(f"Reordered column indexes: {col_labels}")

# now lets get the actual labels
row_actual_labels= data.index[row_labels].tolist()
col_actual_labels= data.columns[col_labels].tolist()

print(f"Reordered row labels: {row_actual_labels}")
print(f"Reordered column labels: {col_actual_labels}")


```

Here you can see that `reordered_ind` gives you the reordered indexes of the initial rows and columns the important part is using the dataframe which you initially used to obtain the index names that correspond to the order that the `clustermap` reordered rows and columns based on the clustering

Now some of you might be asking what if I only want row labels or only column labels or if I am not using a pandas dataframe for the source of data how do I get the labels and also what is `reordered_ind` doing

Let's see what `reordered_ind` is doing with an example where we give it a matrix instead of a dataframe also lets just extract the labels of the columns only

```python

import numpy as np
data2= np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]])

clustermap2 = sns.clustermap(data2,col_cluster=True, row_cluster=False, labels=None)

col_labels_just_indexes = clustermap2.dendrogram_col.reordered_ind


print(f"Reordered column indexes: {col_labels_just_indexes}")
# if you used labels =None this are the indexes that you would need to extract the columns in the data matrix

# to simulate having labels you can use this if you have a list of string with names
example_labels=["col1","col2","col3","col4","col5"]

col_actual_labels_using_example_labels= [example_labels[index] for index in col_labels_just_indexes]

print(f"Reordered column labels using example labels: {col_actual_labels_using_example_labels}")

# but if you are dealing with numpy matrix the easiest would be doing this for instance
reordered_data_columns = data2[:, col_labels_just_indexes]

print (f"Reordered numpy matrix columns based on col reordered indexes: \n {reordered_data_columns}")

```

I usually dont use matrixes directly but that is a way to do it It's important to remember that `reordered_ind` contains the _indices_ of the original rows/columns after the clustering is performed so using the index with original labels either using pandas dataframes or a list that is passed as `labels` in `sns.clustermap` is the way to go

And yes if you only want row labels or column labels just grab what you want from the respective `dendrogram_row` or `dendrogram_col` attributes.

This method works whether you use a DataFrame a numpy array or even a list and using `labels = list_of_strings` in `sns.clustermap` It only requires that you have the original data source available to fetch the actual names or labels by indexes which is standard practice

Now some other common problems people have is using subplots and doing multiple clustermaps and doing stuff with other matplotlib options

Lets do an example with subplots and multiple clustermaps

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # Create figure and axes for subplots

# first clustermap
clustermap1 = sns.clustermap(data.iloc[:, :3], ax=axes[0], col_cluster = True, row_cluster = True,  cmap = "coolwarm")

row_labels_1 = clustermap1.dendrogram_row.reordered_ind
col_labels_1 = clustermap1.dendrogram_col.reordered_ind

row_actual_labels_1= data.index[row_labels_1].tolist()
col_actual_labels_1= data.columns[col_labels_1].tolist()


print(f"Clustermap 1 row labels {row_actual_labels_1}")
print(f"Clustermap 1 col labels {col_actual_labels_1}")


#second clustermap
clustermap2 = sns.clustermap(data.iloc[:, 2:], ax=axes[1], col_cluster = False, row_cluster = True, cmap = "viridis") # Add second clustermap

row_labels_2 = clustermap2.dendrogram_row.reordered_ind
col_labels_2 = clustermap2.dendrogram_col.reordered_ind

row_actual_labels_2= data.index[row_labels_2].tolist()
col_actual_labels_2= data.columns[col_labels_2].tolist()

print(f"Clustermap 2 row labels {row_actual_labels_2}")
print(f"Clustermap 2 col labels {col_actual_labels_2}")


plt.show()

```

In this example notice that I specify the axes to render to this is very important if you are doing subplots and the logic is exactly the same as before using `dendrogram_row` and `dendrogram_col` and `reordered_ind`

Some things to remember: `ClusterGrid` object is your friend not the actual matplotlib axes object so always get the object returned by `sns.clustermap` The reordered indices represent the original order after clustering so they need to be used with the original data source The order of rows and columns will be different from the initial order you provided based on the clustering algorithm that is default or specified when using `sns.clustermap` Always check if you need row labels or columns labels or both of them so you can selectively get the indexes from the dendrogram object accordingly

For deep dives into how dendrograms and clustering work I would check out "The Elements of Statistical Learning" by Hastie Tibshirani and Friedman or "Pattern Recognition and Machine Learning" by Christopher Bishop those are classics and very good resources for statistical methods

I remember a time back in college when I was doing this same exact thing. I spent so much time trying to find those darn labels. I remember I asked my professor and his answer was "have you tried turning it off and on again?"... classic professor humor but at least he gave me a hint to check the properties of the cluster grid object. But yeah after a lot of trial and error I figured it out. I wish stackoverflow was around in those days it would have been a lifesaver.

So yeah that’s pretty much it. Let me know if something else bothers you
