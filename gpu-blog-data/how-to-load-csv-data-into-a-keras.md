---
title: "How to load CSV data into a Keras LSTM model with the correct shape?"
date: "2025-01-30"
id: "how-to-load-csv-data-into-a-keras"
---
The correct shape of input data is critical for an LSTM (Long Short-Term Memory) model in Keras; feeding improperly structured data will result in errors during training. Specifically, LSTMs expect input as a three-dimensional array: `[samples, time steps, features]`. Loading CSV data often requires careful preprocessing to achieve this format before it can be used for model training. I’ve encountered this challenge numerous times in my time building time-series forecasting models, and the devil is always in the details.

Typically, a CSV file presents data in a two-dimensional structure: rows representing different data points, and columns representing different variables (features). The first step is to identify which columns to use as features and which (if any) represent the target variable (for supervised learning). After that, the crucial step is deciding how to structure your *time steps*. This will depend entirely on the nature of your time-series problem.

The "samples" dimension refers to the number of individual sequences you'll feed the model, while "time steps" defines the length of each sequence. The "features" dimension is how many variables you're tracking within each time step. Therefore, achieving the correct data shape involves not only reading the data from the CSV, but also transforming it based on a chosen time window.

Let's break down this process into steps with Python code. We'll assume we have a CSV file named "timeseries.csv" containing time series data.

**1. Reading and Initial Processing**

First, we must read the CSV and convert it into a NumPy array for efficient manipulation. The `pandas` library is commonly used for this:

```python
import pandas as pd
import numpy as np

def load_data(filepath, feature_columns, target_column=None):
    """Loads CSV data, selects features, and optionally target variable."""

    df = pd.read_csv(filepath)
    if target_column:
        target = df[target_column].values
        features = df[feature_columns].values
        return features, target
    else:
        return df[feature_columns].values


# Example usage for a simple time series of two features
filepath = 'timeseries.csv'
feature_columns = ['feature1', 'feature2']
features = load_data(filepath, feature_columns)
print("Initial data shape:", features.shape)

```

In this initial block, we use `pandas` to ingest our CSV, which returns a DataFrame. The `load_data` function can select the specific columns we are interested in as input features for the model, and if required, a target column for training. If a target is included, the function returns both. The first example focuses solely on creating the features array. The output of `features.shape` in this first example will reflect the number of rows in the CSV and the number of selected feature columns. This shape will be 2-dimensional. It's essential to remember that it is not the shape an LSTM requires.

**2. Defining Sequences and Time Steps**

Now, we need to transform our data into sequences with a specific window size (i.e., number of time steps). Let's create a `create_sequences` function to do that:

```python
def create_sequences(data, seq_length):
    """Transforms data into sequences for LSTM input."""
    sequences = []
    for i in range(len(data) - seq_length + 1):
      seq = data[i:(i + seq_length)]
      sequences.append(seq)
    return np.array(sequences)


seq_length = 10  # Example time sequence length
features_seq = create_sequences(features, seq_length)
print("Sequence data shape:", features_seq.shape)

```

The `create_sequences` function slides a window of size `seq_length` along the data, extracting sub-sequences. Each sequence is a series of data points over the defined time window. The crucial step here is converting the list of sequences into a NumPy array, which then results in the desired three dimensional shape. `features_seq.shape` will return (`number_of_sequences, seq_length, number_of_features`). For example, if the original feature set had 100 data points and we set `seq_length=10`, we would get 91 sequences each with 10 time points and however many input features were selected.

**3. Incorporating a Target Variable**

If the dataset contains a target variable for supervised learning, the target data must also be converted into appropriate sequences. Note that the target is often a single value associated with a time sequence.

```python
def create_sequences_with_target(data, target, seq_length):
  """Transforms data into sequences for LSTM input, including targets."""
  sequences = []
  targets = []
  for i in range(len(data) - seq_length):
    seq = data[i:(i + seq_length)]
    sequences.append(seq)
    targets.append(target[i+seq_length])
  return np.array(sequences), np.array(targets)


feature_columns = ['feature1', 'feature2']
target_column = 'target'
features_2, target = load_data(filepath, feature_columns, target_column)
features_seq_2, targets_seq = create_sequences_with_target(features_2, target, seq_length)
print("Target data shape:", targets_seq.shape)
print("Feature data shape:", features_seq_2.shape)

```

This expands the original function to handle a target column from our CSV file. `create_sequences_with_target` generates the same feature sequence as `create_sequences` but additionally constructs a target variable associated with each sequence. Because often we are attempting to predict a target in the future, the target associated with a given sequence is the value of the target column that immediately follows the sequence (i.e. at `i + seq_length` in the original time series). This ensures the output shape for `features_seq_2` will be of shape `(number_of_sequences, seq_length, number_of_features)` while `targets_seq` will have the shape `(number_of_sequences,)`. This shape allows for standard supervised training in Keras models.

**Key Considerations**

*   **Data Normalization:** Before feeding data into an LSTM, it is standard practice to normalize the feature values, e.g., using `sklearn.preprocessing.MinMaxScaler` or `sklearn.preprocessing.StandardScaler`. This ensures that features with larger ranges do not dominate training and can improve convergence. This should be done *before* sequences are generated.

*   **Data Splitting:** Split the data into training, validation, and test sets *after* creating sequences. Random splits of raw data will result in data leakage, as you would have time series data from validation and training sets appearing in the same time sequence. Use a chronological split, separating some percentage of time data for training and the latter portions for validation and testing.

*   **Sequence Length Choice:** The sequence length (time steps) `seq_length` is a hyperparameter that must be tuned based on the specific dataset and problem. Experimentation with different window sizes is critical to model performance. Short window lengths might miss long term dependencies, while long window lengths might make training unnecessarily long.

*   **Statefulness:** For time series that can be broken into separate, independent chunks, the LSTM can be trained as *stateless*. For time-series with long-term dependencies that do not repeat, the LSTM can be trained as *stateful*. Stateful training involves using the final hidden state from each batch to initialize the next batch, which can be important to model long dependencies.

**Resource Recommendations**

For further study, I recommend the following:

*   Books on time series analysis, which often provide more context on data preparation methods for sequential models.
*   Online courses covering LSTMs and recurrent neural networks, which explain the underlying math and network architectures in depth.
*   The official Keras documentation, particularly on the LSTM layer and how to use it for different tasks.
*   Publications on sequence data preprocessing techniques from the machine learning literature.

In summary, the key to loading CSV data into a Keras LSTM model successfully revolves around transforming the typically 2D CSV structure into a 3D tensor: `[samples, time steps, features]`.  Proper sequence generation, choice of sequence length, careful normalization and validation splitting are the keys to effective training.
