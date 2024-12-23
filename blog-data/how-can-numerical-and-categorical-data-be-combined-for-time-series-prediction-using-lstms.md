---
title: "How can numerical and categorical data be combined for time series prediction using LSTMs?"
date: "2024-12-23"
id: "how-can-numerical-and-categorical-data-be-combined-for-time-series-prediction-using-lstms"
---

Alright,  Integrating numerical and categorical data into a time series prediction model using LSTMs is something I’ve dealt with quite extensively over the years, and it's a scenario that pops up far more often than many realize. When I was working on a predictive maintenance project for industrial machinery, we faced a similar challenge, needing to blend sensor readings (numerical) with machine types and maintenance schedules (categorical). It became quickly clear that a naive approach would lead to lackluster results, and we needed to think carefully about how we presented all this information to the LSTM.

The core issue is that LSTMs, by their very nature, operate on numerical data. Categorical variables, being discrete and non-ordinal in most cases, require a preprocessing step before they can be fed into the network. Ignoring this step, or applying it incorrectly, often results in the model struggling to learn meaningful relationships. The most direct way to combine this information is through data preprocessing steps, followed by concatenation. Let me outline this in a systematic manner and then I'll provide a few code snippets to exemplify these concepts.

First, we handle the numerical data. Typically, this involves scaling or normalization. This step ensures that no single feature dominates the learning process solely due to the magnitude of its values. Common methods include min-max scaling (mapping values between 0 and 1) or standardization (converting to a mean of 0 and a standard deviation of 1). The choice often depends on the distribution of your numerical features, and it's something I always experiment with on a validation set.

Second, we address the categorical data. Here, the key transformation is encoding, turning categorical labels into numerical representations. One-hot encoding is common when the categories lack inherent order. Each category becomes a new feature column, where only one column will have a value of '1' for a particular observation. For categorical features with an inherent ordinal relationship, integer encoding is sometimes used, but this must be approached with care, making sure that the assigned numbers are meaningful in that context. It's crucial to consider that the encoding strategy must be appropriate for the specific data; blindly applying one method can lead to poor representation of the underlying relationships.

Once the numerical features have been scaled and the categorical ones encoded, they can be concatenated together into a unified input vector, and this becomes your input data to the LSTM. Remember that each timestep will have this concatenated vector for your sequence. It's not enough just to have all these different kinds of data in a model; they need to be connected logically in the network structure.

I’ve seen various approaches work well, but the simplest architecture often yields strong results: a single LSTM layer followed by dense layers to achieve the final prediction. The number of LSTM units and dense layers is dependent on data complexity, but always start with something reasonable and experiment.

Now, let's move to code examples to show these concepts in practice using python and tensorflow.

**Example 1: Basic Feature Preprocessing and Concatenation**

This first example focuses on the fundamental preprocessing steps before feeding data to a network. This snippet demonstrates how to combine numerical and categorical inputs using one-hot encoding and standardization, using the scikit-learn library for both.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.utils import timeseries_dataset_from_array

# Sample Data - Replace with your actual data
data = pd.DataFrame({
    'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
    'sensor_value': [10, 12, 15, 13, 16],
    'category': ['A', 'B', 'A', 'C', 'B'],
    'target': [20, 22, 25, 24, 27]
})

# Separate numerical and categorical columns
numerical_features = ['sensor_value']
categorical_features = ['category']

# Preprocessing Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Fit the transformer to your training data - assuming 'data' is the training set
preprocessor.fit(data[numerical_features + categorical_features])

# Transform data
processed_data = preprocessor.transform(data[numerical_features + categorical_features])

# Output shape is number of data points x (num features + one hot dimensions)
print(f"Processed data shape: {processed_data.shape}")


sequence_length = 3 # define a length of subsequences
# prepare for sequence-based training. You’ll probably want to do more here based on how you arrange your data.
series_data = timeseries_dataset_from_array(processed_data, data['target'], sequence_length=sequence_length, sequence_stride=1, batch_size=1, shuffle=False)

for element in series_data.take(1):
    x_batch, y_batch = element
    print("Shape of X batch", x_batch.shape)
    print("Shape of y batch", y_batch.shape)

```

This code uses `ColumnTransformer` from `sklearn` to apply different transformations based on column type. This results in a single, combined processed data, which you can then reshape into sequences suitable for LSTMs using TensorFlow's `timeseries_dataset_from_array` function or a manual approach that is appropriate for your data.

**Example 2: Building a Basic LSTM Model**

This example builds a simple LSTM model to receive the combined data. This is where the preprocessed data becomes input to the network.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Define input parameters - use the output from the preprocessor
input_shape = (sequence_length, processed_data.shape[1])

model = Sequential([
    Input(shape = input_shape),
    LSTM(50, activation='relu', return_sequences=False),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

print(model.summary())

# Placeholder training. Actual training needs to account for splitting your dataset and using epochs.
model.fit(series_data, epochs=10)


# Placeholder Prediction.
example_predict =  series_data.take(1)
for x_batch, y_batch in example_predict:
    prediction = model.predict(x_batch)
    print("Shape of prediction:", prediction.shape)
    print("Prediction:", prediction)

```

This example constructs a simple LSTM network with an input shape derived from the preprocessed data. The `return_sequences=False` argument in the LSTM layer is appropriate here since we are predicting a single output value at the end of each sequence. The dense layer then completes the prediction task.

**Example 3: More Sophisticated Encoding**

This example shows how to extend the encoding to consider ordinal relationships with custom transformers (if your particular situation warrants it).

```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.utils import timeseries_dataset_from_array

class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.map(self.mapping)

# Sample Data
data = pd.DataFrame({
    'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
    'sensor_value': [10, 12, 15, 13, 16],
    'category': ['low', 'medium', 'low', 'high', 'medium'],
    'target': [20, 22, 25, 24, 27]
})

numerical_features = ['sensor_value']
categorical_features = ['category']

# Define an ordinal mapping - Use carefully, only when data contains real ordinal relationships.
ordinal_mapping = { 'low': 0, 'medium': 1, 'high': 2 }

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('ord', OrdinalEncoder(mapping=ordinal_mapping), categorical_features)
    ])

# Fit and transform - the same as before.
preprocessor.fit(data[numerical_features + categorical_features])
processed_data = preprocessor.transform(data[numerical_features + categorical_features])

print(f"Processed data shape: {processed_data.shape}")

sequence_length = 3
series_data = timeseries_dataset_from_array(processed_data, data['target'], sequence_length=sequence_length, sequence_stride=1, batch_size=1, shuffle=False)

for element in series_data.take(1):
    x_batch, y_batch = element
    print("Shape of X batch", x_batch.shape)
    print("Shape of y batch", y_batch.shape)
```

This shows you how to use a custom transformer to implement ordinal encoding, when it is appropriate to do so.

This is a solid basis for combining numerical and categorical data. For further study, I suggest delving into "Deep Learning with Python" by François Chollet, for the fundamental understanding of LSTMs and preprocessing methods. Additionally, the scikit-learn documentation provides a fantastic explanation of `ColumnTransformer` and its usage. Furthermore, research papers focusing on 'time series forecasting with mixed data types' are also useful, but you will have to tailor the search based on specifics of your own data and application area. Specifically, look into the concepts of 'embedding layers' when categorical data becomes high cardinality.

Remember that this is a starting point. Model architecture, learning rates, regularization, and even data augmentation techniques can make or break a system like this. The key is to experiment, monitor your results, and understand the specific characteristics of your dataset.
