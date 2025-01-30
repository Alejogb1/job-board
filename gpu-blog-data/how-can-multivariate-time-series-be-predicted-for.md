---
title: "How can multivariate time series be predicted for non-continuous data?"
date: "2025-01-30"
id: "how-can-multivariate-time-series-be-predicted-for"
---
Multivariate time series forecasting, particularly when dealing with non-continuous data, presents unique challenges compared to its continuous counterpart. The inherent discreteness and often categorical nature of the data necessitate careful consideration of model selection and data preprocessing techniques. I've encountered this firsthand when analyzing sensor data from industrial machinery, where various discrete states (e.g., operational modes, error codes) coexisted with continuous readings like temperature and vibration. This complexity demanded a shift from standard regression-based approaches.

The primary difficulty stems from the fact that traditional time series models, like ARIMA or Exponential Smoothing, typically assume numerical continuity. Discrete data, on the other hand, lacks the smooth transitions these models rely on. Furthermore, multivariate scenarios introduce interdependencies between these often heterogeneous data streams. Ignoring these dependencies can lead to significantly inaccurate predictions. Therefore, appropriate preprocessing and model selection become paramount.

Before modeling, the data requires transformation. For categorical variables, one-hot encoding or integer encoding is necessary. One-hot encoding expands a single categorical feature into multiple binary features, preserving information but potentially increasing dimensionality. Integer encoding assigns a unique integer to each category, reducing dimensionality but implies an ordinal relationship, which might not exist. When dealing with a mix of continuous and categorical data, a common strategy is to keep the continuous data as is, while one-hot or integer encode the categorical variables, resulting in a hybrid feature space. This step is crucial for feeding the data into machine learning algorithms. Another crucial aspect of preprocessing is handling missing values. If missing values are prevalent, imputation techniques, like mode imputation for categorical data or linear interpolation for continuous data, must be implemented to avoid losing critical temporal information. Proper scaling of continuous data is also required if various ranges of continuous data are used, to avoid issues with model training and optimization.

For modeling, I've found that Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, and variations of sequence-to-sequence models, are particularly effective when handling time dependencies in non-continuous multivariate data. While these architectures are generally powerful with time-series data, their effectiveness depends heavily on the specific application, hyperparameters chosen, and data-preprocessing strategies.

Here’s an illustration using Python with Keras, a high-level neural network API:

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def create_lstm_model(input_shape, num_categories):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2), # Add dropout layer for regularization
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(num_categories, activation='softmax') # Use softmax for multi-class
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage
time_steps = 10
num_features = 3  # two continuous, one categorical
num_categories = 4  # Number of categories for the discrete variable

# Sample data: time_series of (continuous1, continuous2, categorical)
# Generate synthetic data
np.random.seed(42) # for reproducibility
X = np.random.rand(100, time_steps, num_features)
categorical_data = np.random.randint(0, num_categories, size=(100, time_steps, 1))
X[:, :, -1] = categorical_data[:,:,0]

# Example conversion of categorical into one-hot
def one_hot_encode(data, num_categories):
    encoded = np.zeros((data.shape[0], data.shape[1], num_categories))
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
            encoded[i, j, data[i,j].astype(int)] = 1
    return encoded

X_cat_encoded = one_hot_encode(X[:,:,-1:], num_categories)
X = np.concatenate((X[:,:,:num_features-1], X_cat_encoded), axis=2)

# Create corresponding labels
Y = np.random.randint(0, num_categories, size=(100))
Y_encoded = np.zeros((100, num_categories))
for i, val in enumerate(Y):
    Y_encoded[i, val] = 1

input_shape = (time_steps, num_features -1 + num_categories)

model = create_lstm_model(input_shape, num_categories)
model.fit(X, Y_encoded, epochs=10, verbose = 0)
```

This code defines a basic LSTM network configured to handle multivariate time series input, including both continuous and one-hot encoded categorical features. The final dense layer with a softmax activation function is suitable for multi-class classification of the discrete variable, common for prediction of categorical outcomes. The dropout layers help with preventing overfitting. The key feature is that the last feature of the input has been one-hot encoded, increasing the size of the feature dimension.

Furthermore, the use of sequence-to-sequence models can be important when the forecasting target is also a time series, as opposed to single-value prediction. These models typically employ an encoder-decoder architecture, where the encoder processes the input time series and the decoder generates the predicted sequence. Below is an example of this setup using a single LSTM layer as an encoder, and another for a decoder.

```python
def create_seq2seq_model(input_shape, output_shape):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    encoder_lstm = layers.LSTM(64, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = layers.Input(shape= (1,output_shape[-1]))
    decoder_lstm = layers.LSTM(64, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = layers.Dense(output_shape[-1], activation='softmax') # output
    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (time_steps, num_features -1 + num_categories)
output_shape = (1, num_categories)

model = create_seq2seq_model(input_shape, output_shape)

# Create Dummy data
# Sample data: time_series of (continuous1, continuous2, categorical)
np.random.seed(42) # for reproducibility
X = np.random.rand(100, time_steps, num_features)
categorical_data = np.random.randint(0, num_categories, size=(100, time_steps, 1))
X[:, :, -1] = categorical_data[:,:,0]

# Example conversion of categorical into one-hot
X_cat_encoded = one_hot_encode(X[:,:,-1:], num_categories)
X = np.concatenate((X[:,:,:num_features-1], X_cat_encoded), axis=2)
# Create corresponding labels
Y = np.random.randint(0, num_categories, size=(100, 1))
Y_encoded = np.zeros((100, 1, num_categories))
for i, val in enumerate(Y):
    Y_encoded[i, 0, val] = 1

decoder_input = np.zeros((100, 1, num_categories))

model.fit([X, decoder_input], Y_encoded, epochs=10, verbose = 0)

```

This example sets up the sequence-to-sequence learning with two inputs. Note that it uses teacher forcing for training. In this case, a decoder input is defined as a zero array with the correct shape. In practice, the decoder input would consist of the ground truth, as this allows the model to learn effectively. The crucial aspect here is the explicit separation between the input and output time series. The output in this scenario is a sequence of probabilities of a categorical output.

Finally, in situations where computational resources are limited or interpretability is a primary concern, simpler models such as Random Forests or Gradient Boosting Machines (GBM) can be effective, provided they are appropriately configured. These models generally require feature engineering to be effective with temporal data.  For this, lagging features that summarize the recent history of time series data are common and can be included as static features.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample time-series data generation
np.random.seed(42)
n_samples = 100
time_steps = 10
num_features = 3
num_categories = 4

# Generate Synthetic data
X = np.random.rand(n_samples, time_steps, num_features)
categorical_data = np.random.randint(0, num_categories, size=(n_samples, time_steps, 1))
X[:, :, -1] = categorical_data[:,:,0]

# Example conversion of categorical into one-hot
def one_hot_encode(data, num_categories):
    encoded = np.zeros((data.shape[0], data.shape[1], num_categories))
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
            encoded[i, j, data[i,j].astype(int)] = 1
    return encoded

X_cat_encoded = one_hot_encode(X[:,:,-1:], num_categories)
X = np.concatenate((X[:,:,:num_features-1], X_cat_encoded), axis=2)

# Create labels
Y = np.random.randint(0, num_categories, size=(n_samples))

# Feature engineering
def create_lagged_features(data, num_lags):
    lagged_features = []
    for i in range(1, num_lags + 1):
      lagged_features.append(np.roll(data, shift=i, axis=1))
    return np.concatenate(lagged_features, axis=2)

num_lags = 3
X_lagged = create_lagged_features(X, num_lags)
X_lagged_reshaped = X_lagged[:, time_steps-1, :] # Take last time step as input to the ML model.

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_lagged_reshaped, Y, test_size=0.2, random_state=42)

# Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```
Here, the time-series data has been transformed into a lagged feature set, using the past `num_lags` time steps, and passed to a Random Forest classifier. While more basic than the earlier LSTM and sequence-to-sequence models, with carefully engineered features, this technique can prove to be a reasonable approach to the prediction of multivariate discrete data. The accuracy calculation demonstrates how the model may be evaluated.

For more information on time series analysis, I would recommend looking into textbooks focusing on statistical forecasting. Books on machine learning can provide a good theoretical foundation for these modeling techniques. For more practical application, publications on neural network model implementation are invaluable, particularly when employing RNNs or sequence-to-sequence models. Finally, studying resources on feature engineering will be useful for building better time-series forecasting models, especially when simple models such as Random Forests or GBMs are utilized.
