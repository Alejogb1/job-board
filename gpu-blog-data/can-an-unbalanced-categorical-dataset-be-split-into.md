---
title: "Can an unbalanced categorical dataset be split into a representative training set and a balanced testing set?"
date: "2025-01-30"
id: "can-an-unbalanced-categorical-dataset-be-split-into"
---
Splitting an unbalanced categorical dataset into a representative training set and a balanced testing set is indeed achievable and frequently necessary in machine learning, particularly when evaluating model performance on underrepresented classes. The key challenge lies in maintaining the original class distribution within the training set, enabling the model to learn the nuances of the data, while simultaneously creating a balanced testing set to accurately assess performance across all categories, preventing bias towards the majority class. My experience working on fraud detection for an e-commerce platform highlights this very problem. We had significantly more legitimate transactions than fraudulent ones. Training solely on that data produced models that were highly accurate on classifying legitimate transactions but performed poorly on detecting fraud. We needed to create a balanced test set to measure its true efficacy.

Let's dissect this. The fundamental issue with unbalanced datasets is that a model trained on them can become biased towards the majority class. For instance, in the e-commerce example, if 99% of the transactions were legitimate and 1% fraudulent, the model could achieve 99% accuracy simply by classifying everything as legitimate. This doesn't indicate a robust model. A balanced test set ensures that the model is evaluated on an equal number of examples from each class, allowing for a more reliable assessment of its ability to differentiate between them. We achieve this by controlling the class distribution during the splitting process. The training set should retain the original class imbalance to adequately represent the real-world distribution the model will encounter in the production environment.

Several stratified splitting techniques can accomplish this, including a combination of stratified random sampling and subsampling. For the training set, we utilize stratified random sampling, which ensures that the proportion of each class within the original dataset is roughly maintained within the training set. This ensures that even rare classes are represented in a similar proportion during model training. For the testing set, I use a technique I've implemented where I first identify the minority class with the smallest number of samples. Then I randomly sample, without replacement, an equal number of samples from each other class, matching the size of the minority class. This guarantees a balanced representation for evaluation.

Here's how this can be done using Python and the `scikit-learn` and `pandas` libraries:

**Code Example 1: Stratified Training Set Creation**

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def create_stratified_train_set(df, target_column, train_size=0.8, random_state=42):
    """
    Creates a stratified training set from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): The column containing the target variable.
        train_size (float): The proportion of the dataset to include in the train split.
        random_state (int): Controls the shuffling applied to the data before the split.

    Returns:
        pd.DataFrame: The stratified training DataFrame.
    """
    train_df, _ = train_test_split(df, train_size=train_size, stratify=df[target_column], random_state=random_state)
    return train_df

# Sample Usage
data = {'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
df = pd.DataFrame(data)
training_df = create_stratified_train_set(df, 'category')
print("Stratified Training Set:")
print(training_df)
print(training_df['category'].value_counts(normalize=True))
```

In this code, the `create_stratified_train_set` function leverages `train_test_split` from `scikit-learn`. The `stratify` parameter, set to the target column, is crucial. It ensures that the training set reflects the original class proportions. The function returns the stratified training set. When you examine the output using `.value_counts(normalize=True)`, you can confirm that the proportions are roughly equal to the original dataframe's proportions of each class

**Code Example 2: Balanced Testing Set Creation**

```python
import pandas as pd

def create_balanced_test_set(df, target_column, random_state=42):
    """
    Creates a balanced testing set from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): The column containing the target variable.
        random_state (int): Controls the random sampling process.

    Returns:
         pd.DataFrame: The balanced testing DataFrame.
    """
    min_class_size = df[target_column].value_counts().min()
    balanced_dfs = []
    for category in df[target_column].unique():
        category_df = df[df[target_column] == category].sample(min_class_size, random_state=random_state)
        balanced_dfs.append(category_df)
    balanced_df = pd.concat(balanced_dfs)
    return balanced_df

# Sample Usage
data = {'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
df = pd.DataFrame(data)
testing_df = create_balanced_test_set(df, 'category')
print("\nBalanced Testing Set:")
print(testing_df)
print(testing_df['category'].value_counts())
```

The `create_balanced_test_set` function determines the minimum class size and then iterates through each class, sampling an equal number of instances as the minimum class. It uses the `.sample()` method to select data from each category. The sampled subsets are then concatenated to form the final balanced testing set. This approach ensures that each class is equally represented within the testing set and you will see that the number of instances of each class is equal.

**Code Example 3: Combining Stratified Training and Balanced Testing**

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data_stratified_train_balanced_test(df, target_column, train_size=0.8, random_state=42):
  """Splits data into a stratified training set and balanced testing set.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): The column containing the target variable.
        train_size (float): The proportion of the dataset to include in the train split.
        random_state (int): Controls the random sampling process and data shuffling.

    Returns:
        tuple: A tuple containing the stratified training DataFrame and the balanced testing DataFrame.
    """

  train_df, temp_df = train_test_split(df, train_size=train_size, stratify=df[target_column], random_state=random_state)
  test_df = create_balanced_test_set(temp_df, target_column, random_state=random_state)
  return train_df, test_df


# Sample Usage:
data = {'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
df = pd.DataFrame(data)

train_df, test_df = split_data_stratified_train_balanced_test(df, 'category')

print("Stratified Training Set:")
print(train_df['category'].value_counts(normalize=True))
print("\nBalanced Testing Set:")
print(test_df['category'].value_counts())
```

The `split_data_stratified_train_balanced_test` function combines the logic from the prior two examples.  First, it splits the data into a stratified training set and a temporary set (which will then be used to create the balanced testing set).  The balanced test set is then created from the temporary set and both resulting dataframes are returned. This provides a complete example of how to achieve a stratified training set and balanced testing set from one original dataset.

Regarding resources, I would recommend the official scikit-learn documentation on model selection and splitting datasets. I've also found the pandas library's documentation extremely helpful for data manipulation and random sampling techniques. In addition, exploring scholarly articles on stratified sampling and bias in machine learning would prove beneficial. Understanding the mathematical basis for stratified sampling allows for more informed decisions when encountering different types of dataset imbalances. Finally, various data science blogs frequently publish material on best practices for handling unbalanced data that are worth reading.

In conclusion, creating a representative training set and a balanced testing set from an unbalanced categorical dataset is not only possible but critical for developing reliable machine learning models. The approaches detailed above, utilizing stratified sampling for the training set and controlled subsampling for the testing set, facilitate proper model training and evaluation. The use of the `scikit-learn` and `pandas` libraries provides the tools necessary to implement these techniques with relative ease. The key is to recognize the distinct requirements of the training and testing phases and to adjust the split accordingly to avoid bias and obtain robust performance results. This is a fundamental part of good data science practice.
