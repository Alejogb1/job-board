---
title: "How can I perform stratified train-test split in a DataFrame to maintain equal class proportions?"
date: "2025-01-30"
id: "how-can-i-perform-stratified-train-test-split-in"
---
Maintaining class proportions during train-test splits is critical for building robust classification models, especially when dealing with imbalanced datasets. If a standard random split is employed, the resulting training and test sets may not accurately represent the overall distribution of classes, leading to biased model performance. My experience has shown that stratified splitting is often a crucial preprocessing step to ensure reliable evaluation of classifier performance. I will detail how to perform this operation, specifically within the context of working with dataframes, and provide practical code illustrations to demonstrate this concept.

The core idea behind stratified splitting is to divide the dataset in a way that preserves the original class distribution within both the training and test subsets. This involves partitioning each class individually, aiming for proportional representation in the generated sets. For example, if your dataset has 70% class 'A' and 30% class 'B,' a stratified split will attempt to retain these same proportions, or as close as possible, in the train and test subsets. It avoids accidentally ending up with, say, a training set with a far lower representation of class 'B'.

The process, when using common data manipulation libraries such as pandas in Python, generally relies on grouping the data by the class label and then randomly selecting a subset of each group for training. The remaining samples form the test dataset. The crucial function here is the splitting mechanism, which ensures a proportional representation is maintained during the partitioning. The precise implementation will vary depending on the toolkit used, but the underlying logic remains constant.

Here are three code examples that illustrate how stratified splitting can be implemented using Python with pandas and scikit-learn.

**Example 1: Manual Implementation with pandas `groupby`**

This example showcases a manual approach utilizing pandas to achieve stratified splitting. It's more verbose but provides clarity about the underlying mechanics.

```python
import pandas as pd
import numpy as np

def stratified_split_pandas(df, target_column, test_size=0.2, random_state=None):
    """Performs stratified train-test split using pandas groupby."""

    if random_state:
        np.random.seed(random_state)

    train_df_list = []
    test_df_list = []

    for _, group in df.groupby(target_column):
       n_test = int(len(group) * test_size)
       test_indices = np.random.choice(group.index, n_test, replace=False)
       test_df_list.append(group.loc[test_indices])
       train_indices = group.index.difference(test_indices)
       train_df_list.append(group.loc[train_indices])


    train_df = pd.concat(train_df_list)
    test_df = pd.concat(test_df_list)

    return train_df, test_df

# Sample DataFrame
data = {'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])}
df = pd.DataFrame(data)

train_df, test_df = stratified_split_pandas(df, 'target', test_size=0.3, random_state=42)

print("Train set class distribution:\n", train_df['target'].value_counts(normalize=True))
print("\nTest set class distribution:\n", test_df['target'].value_counts(normalize=True))

```

In this example, the `stratified_split_pandas` function first seeds the random number generator if a `random_state` is provided to guarantee reproducible results. Then, it groups the dataframe by the target column, essentially dividing it into subsets, each corresponding to a single class. For every group, the function calculates the number of samples needed for the test set based on `test_size`. Then random samples from each class are selected for the test set and the remainder is used for the training set. After each group is processed, the training and test sets are concatenated back to obtain the final split DataFrames. Finally, the code demonstrates the application on a sample DataFrame and prints the resulting class distributions to confirm that they remain close to the original dataset proportions.

**Example 2: Using `scikit-learn` `train_test_split` with the `stratify` argument.**

This example utilizes scikit-learn's built-in `train_test_split` function. This provides a more concise method, but requires an understanding of how to extract the features and target variables from the DataFrame. This is my preferred method due to its simplicity and flexibility.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Sample DataFrame
data = {'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])}
df = pd.DataFrame(data)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print("Train set class distribution:\n", train_df['target'].value_counts(normalize=True))
print("\nTest set class distribution:\n", test_df['target'].value_counts(normalize=True))
```

Here, we utilize the `train_test_split` function from scikit-learn. Crucially, the `stratify` parameter is set to the target variable 'y'. This signals the function to perform stratified splitting. Before calling the function, the features (`X`) and target variable (`y`) are extracted from the DataFrame. After splitting, training and testing sets containing both features and target variables are concatenated and printed. This example illustrates that a single function call can replace significant coding from example 1 with similar outcome while providing flexibility to specify test sizes.

**Example 3: Using pandas `sample` with explicit proportions.**

This example uses the `sample` function along with calculated target class distribution. While manual, it offers granular control over the proportions and potentially addresses datasets with very small class sizes.

```python
import pandas as pd
import numpy as np

def stratified_split_sample(df, target_column, test_size=0.2, random_state=None):
  """Performs stratified train-test split using pandas sample."""

  if random_state:
    np.random.seed(random_state)

  train_df_list = []
  test_df_list = []

  class_proportions = df[target_column].value_counts(normalize=True)

  for class_label, proportion in class_proportions.items():
    class_df = df[df[target_column] == class_label]
    n_test = int(len(class_df) * test_size)

    test_sample = class_df.sample(n=n_test, random_state=random_state)
    train_sample = class_df.drop(test_sample.index)

    test_df_list.append(test_sample)
    train_df_list.append(train_sample)


  train_df = pd.concat(train_df_list)
  test_df = pd.concat(test_df_list)

  return train_df, test_df


# Sample DataFrame
data = {'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])}
df = pd.DataFrame(data)

train_df, test_df = stratified_split_sample(df, 'target', test_size=0.3, random_state=42)

print("Train set class distribution:\n", train_df['target'].value_counts(normalize=True))
print("\nTest set class distribution:\n", test_df['target'].value_counts(normalize=True))
```
This third example provides more direct control over proportions. The `stratified_split_sample` function first calculates the class distributions. It iterates over each class and samples the test set according to provided `test_size`. The remaining instances are then added to the train set and then returns the concatenated sets. The resulting class distributions are then printed confirming the stratification. This approach is useful when `test_size` is not straightforward, or when you need very specific control over the splits.

In choosing the correct approach, I consider two factors: First, libraries, as `scikit-learn` is well-optimized. I have observed that `train_test_split` performs fast and reliable splits, provided that the data is well-structured. Second, it is essential to understand the underlying logic. The examples showcase three different approaches, including implementing the logic manually. Manual implementation can be a valuable exercise as it allows the user to clearly understand and implement the stratified splitting in specific use cases. When a black-box function like scikit learn is used, the inner workings of stratification are hidden, which may not be ideal when debugging or when a more complicated sampling approach is needed.

As a resource, I strongly recommend studying the official pandas and scikit-learn documentation. These provide detailed information about the capabilities of each function, including their potential limitations. I would also recommend reading through case studies from machine learning practitioners online, which will provide more specific scenarios with possible best practices. Furthermore, any introductory machine learning text that covers data preprocessing, especially for classification problems, will provide a detailed explanation.
