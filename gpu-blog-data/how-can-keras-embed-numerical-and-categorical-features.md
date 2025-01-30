---
title: "How can Keras embed numerical and categorical features for RNNs?"
date: "2025-01-30"
id: "how-can-keras-embed-numerical-and-categorical-features"
---
A recurrent neural network (RNN), specifically designed for sequential data, requires its inputs to be represented as a numerical sequence. This presents a challenge when dealing with datasets containing both numerical and categorical features, as these data types are inherently different. My experience working with time series forecasting and natural language processing has repeatedly underscored this critical data preprocessing step. Simply passing raw categorical labels or numerical values will not produce effective training outcomes. We must employ techniques to transform these features into a format amenable to RNN processing. The primary method for achieving this involves utilizing embedding layers and scaling techniques.

**Numerical Feature Embedding and Scaling**

Numerical features, while already in a numerical format, often require scaling before being fed into an RNN. This is crucial for stabilizing the training process and ensuring that no feature disproportionately influences learning due to differing scales. For instance, a feature representing the age of a customer (values ranging from 18 to 80) has a significantly smaller magnitude than one representing annual income (values ranging from 20,000 to 200,000). Directly incorporating such disparate values into the network will likely lead to difficulties in gradient descent and converge poorly.

Scaling techniques are applied independently to each numerical feature. Common approaches include:

*   **Min-Max Scaling (Normalization):** This method rescales the data to a specific range, typically between 0 and 1. It is calculated as: `(x - min(x)) / (max(x) - min(x))`
*   **Standard Scaling (Z-score Normalization):** This approach scales the data to have a mean of 0 and a standard deviation of 1. It is calculated as: `(x - mean(x)) / std(x)`
*   **Robust Scaling:** Similar to standard scaling, but it uses the median and interquartile range, making it less sensitive to outliers.

Choosing the appropriate scaling method depends on the distribution of the numerical feature. Normalization is suitable for bounded data, while standard scaling is often effective when data distribution is close to a Gaussian. Robust scaling is a solid choice when dealing with datasets that contain significant outliers, which I've encountered frequently in real-world datasets.

After scaling, numerical features can be directly fed into an RNN by passing each time step as part of the input sequence. The crucial aspect is to ensure that the time dimension is correctly shaped, typically requiring reshaping or padding operations.

**Categorical Feature Embedding**

Categorical features, representing discrete values like product type, city, or weekdays, need to be converted to a numerical representation, often through the use of embedding layers. The key distinction between categorical features and numerical ones is the lack of inherent numerical relationship within the categories. Treating categories as simple integers (like 0, 1, 2...) can introduce spurious order-based relationships which have no meaning in context of your data, leading to inadequate performance in your model.

Embedding layers address this issue. An embedding layer maps each unique category value to a dense vector of a specified size. For example, consider a categorical feature representing the day of the week with seven possible values. We can initialize an embedding layer to map each of these seven values to a vector of, say, size 16. Each category now has a unique 16-dimensional representation. These vectors are learned during model training, allowing the network to capture relationships between categories in the representation space itself.

The embedding layer is typically the first processing step for a categorical feature before it is combined or concatenated with other features. It transforms the discrete categorical values into a dense, learned numerical form that can be further processed by subsequent layers within the RNN.

**Code Examples and Commentary**

Below, I provide three Python code examples using Keras to illustrate the concepts of numerical scaling, categorical embedding, and the preparation of a data stream for an RNN.

**Example 1: Numerical Feature Scaling and RNN Input Preparation**

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape

# Sample numerical time-series data
numerical_data = np.random.rand(100, 10, 1)  # 100 sequences of length 10 with 1 feature each

# Scaling with MinMaxScaler
scaler = MinMaxScaler()
scaled_data = np.array([scaler.fit_transform(seq) for seq in numerical_data])

# Define input layer, reshape for LSTM and output layer of network
input_layer = Input(shape=(10, 1))
reshaped_input = Reshape((10,1))(input_layer) # Explicit reshaping if necessary
lstm_layer = LSTM(32)(reshaped_input)
output_layer = Dense(1)(lstm_layer)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='mse')

#Dummy training data, passing already scaled data
dummy_labels = np.random.rand(100,1)
model.fit(scaled_data, dummy_labels, epochs=2) # Training the model

# Print a summary for verification
model.summary()

```

This example showcases the use of MinMaxScaler to scale numerical data before it is passed through an LSTM network. The reshape layer ensures that the input shape is compatible with the LSTM. The input shape and subsequent layers must reflect the temporal information that the network will use.

**Example 2: Categorical Feature Embedding**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, LSTM, Dense
from tensorflow.keras.models import Model

# Sample categorical data (sequence of indices)
categorical_data = np.random.randint(0, 5, size=(100, 10)) # 100 sequences of length 10, max category is 5

# Number of unique categories
num_categories = 5

# Embedding dimension
embedding_dim = 8

#Define embedding input and embedding layer
input_layer = Input(shape=(10,))
embedding_layer = Embedding(input_dim=num_categories, output_dim=embedding_dim)(input_layer)

# Flatten to provide an input for a Dense layer or reshaping before processing by recurrent networks
flattened_embedding = Flatten()(embedding_layer)
reshaped_embedding = tf.reshape(embedding_layer, shape=(-1, 10, 8))

# LSTM after embedding
lstm_layer = LSTM(32)(reshaped_embedding)
output_layer = Dense(1)(lstm_layer)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse')

# Training with dummy labels
dummy_labels = np.random.rand(100,1)
model.fit(categorical_data, dummy_labels, epochs=2)

# Model summary for verification
model.summary()
```

This example demonstrates how to use an Embedding layer to convert categorical sequences into dense vector representations. The `input_dim` is the number of unique categories and `output_dim` is the size of the embedding vectors. Note the explicit reshaping of the tensor to be compatible with an LSTM layer.

**Example 3: Combining Numerical and Categorical Features**

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Reshape
from tensorflow.keras.models import Model

# Sample numerical and categorical data
numerical_data = np.random.rand(100, 10, 1)
categorical_data = np.random.randint(0, 5, size=(100, 10))
num_categories = 5
embedding_dim = 8

# Scaling numerical data
scaler = MinMaxScaler()
scaled_numerical_data = np.array([scaler.fit_transform(seq) for seq in numerical_data])

# Inputs definitions
numerical_input = Input(shape=(10, 1))
categorical_input = Input(shape=(10,))

# Embedding layer for categorical features
embedding_layer = Embedding(input_dim=num_categories, output_dim=embedding_dim)(categorical_input)
reshaped_embedding = tf.reshape(embedding_layer, shape=(-1, 10, 8))

# Reshape numerical data
reshaped_numerical_input = Reshape((10,1))(numerical_input)

# Concatenate numerical and embedded categorical features
concatenated_input = Concatenate(axis=2)([reshaped_numerical_input, reshaped_embedding])


# LSTM layer
lstm_layer = LSTM(32)(concatenated_input)
output_layer = Dense(1)(lstm_layer)

# Create the model
model = Model(inputs=[numerical_input, categorical_input], outputs=output_layer)
model.compile(optimizer='adam', loss='mse')

#Training
dummy_labels = np.random.rand(100,1)
model.fit([scaled_numerical_data, categorical_data], dummy_labels, epochs=2)

model.summary()
```

This final example illustrates the combination of scaled numerical and embedded categorical features. These two are first processed separately and then concatenated before being passed into the LSTM. This is a standard architecture when having multiple data types associated with a time series.

**Resource Recommendations**

For a more in-depth exploration, I recommend consulting resources focusing on data preprocessing for machine learning. Texts covering feature engineering and representation learning provide a solid theoretical background. Publications and documentation related to TensorFlow and Keras are vital for hands-on implementations and are updated regularly to reflect best practices. Additionally, specific modules related to scikit-learn preprocessing are highly relevant for scaling and other numerical data preparation tasks. Investigating use-cases within natural language processing can provide further insights in practical usage of embedding layers, especially when you're dealing with sequential categorical input like words in sentences.
