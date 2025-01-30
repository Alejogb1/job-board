---
title: "How can TensorFlow be used to predict time series from non-continuous data?"
date: "2025-01-30"
id: "how-can-tensorflow-be-used-to-predict-time"
---
Time series prediction using TensorFlow often encounters challenges when dealing with non-continuous data, primarily because the temporal relationships are not as direct or readily apparent as in, say, stock prices or sensor readings sampled at regular intervals. This discontinuity could stem from sporadically measured observations or data gaps. Building predictive models in such scenarios requires careful feature engineering and selection of appropriate network architectures. My experience in developing predictive maintenance models for industrial machinery, where sensor readings are often intermittent due to network issues or power limitations, has directly informed my approach.

The core issue is the traditional reliance of recurrent neural networks (RNNs), specifically LSTMs (Long Short-Term Memory networks) or GRUs (Gated Recurrent Units), on continuous, sequential input. These networks are designed to process a stream of data points where each point is assumed to be closely related in time to its immediate neighbors. When data is non-continuous, applying these directly can lead to misleading results, as the network might incorrectly interpret the gaps or irregular sampling as significant fluctuations. Therefore, a critical step before feeding data into an RNN or similar architecture is to convert the non-continuous data into a format suitable for sequential modeling. This preprocessing typically involves several key stages: handling missing values, creating meaningful temporal features, and potentially resampling the data to a consistent frequency.

Let's delve into a concrete scenario: imagine we're monitoring machine tool wear based on operational parameters that are recorded only when the tool is actively cutting. This results in intermittent data points—sometimes many readings within a short period, sometimes no data for extended durations. Directly applying a standard LSTM would be problematic.

First, consider **missing value imputation**. Simple techniques such as forward-fill (copying the last observed value) or backward-fill could be used, but these are often inadequate for situations with long data gaps, potentially skewing the temporal dynamics. More sophisticated approaches like interpolating values between known points or using machine learning models to impute missing data are sometimes necessary. However, for illustrative purposes and keeping with simplicity, forward fill can be sufficient as a first approach. Following imputation, we must then address temporal feature engineering.

One effective strategy is to convert non-continuous time series data into a series of fixed-length sequences. This involves determining a fixed time window (e.g., one hour, one day) and partitioning the original data into segments of that window length. Within each window, we can derive statistical features such as the mean, median, standard deviation, minimum, and maximum of the observed values. Additionally, we can calculate the duration of non-missing data within that window as an additional feature. These are no longer discrete points but rather time-based aggregations. This transforms the data into a time-series suitable for an RNN.

Let’s explore the first code example. This snippet demonstrates how to preprocess data by creating these windowed statistical features.

```python
import numpy as np
import pandas as pd

def create_windowed_features(data, time_col, feature_col, window_size='1H'):
    """
    Transforms a DataFrame with non-continuous time series into a DataFrame with windowed features.

    Args:
        data (pd.DataFrame): DataFrame with time series data.
        time_col (str): Name of the time column.
        feature_col (str): Name of the feature column.
        window_size (str): String specifying the window size (e.g., '1H' for one hour).

    Returns:
        pd.DataFrame: DataFrame with windowed features.
    """
    data[time_col] = pd.to_datetime(data[time_col])
    data.sort_values(by=time_col, inplace=True)
    data.set_index(time_col, inplace=True)
    
    resampled_data = data.resample(window_size).agg({
        feature_col: ['mean', 'median', 'std', 'min', 'max', lambda x: x.count() / (x.notna().sum() if x.notna().sum() > 0 else 1) ]
    })
    resampled_data.columns = ['_'.join(col).strip() for col in resampled_data.columns.values]
    resampled_data.reset_index(inplace=True)
    resampled_data.rename(columns = { resampled_data.columns[0] : time_col}, inplace = True)
    resampled_data.rename(columns = { resampled_data.columns[-1] : feature_col + '_coverage'}, inplace = True)
    return resampled_data

# Example Usage
data = {'time': ['2024-01-01 00:00:00', '2024-01-01 00:15:00', '2024-01-01 02:00:00', '2024-01-01 02:30:00', '2024-01-01 04:30:00', '2024-01-01 05:00:00'],
        'sensor_reading': [10, 12, 15, 14, 18, 19]}
df = pd.DataFrame(data)
windowed_df = create_windowed_features(df, 'time', 'sensor_reading')
print(windowed_df)

```

This function, `create_windowed_features`, takes a Pandas DataFrame with a time column and a feature column, as well as a window size, and aggregates the data within each time window to derive statistical measures. The results from the aggregation (mean, median, std, min, max and coverage) are returned in a new DataFrame. The `coverage` column represents the fraction of the time window that had non-missing data (this is important since if no data was recorded in an entire window, its aggregation stats are meaningless). Notice the use of the lambda function for coverage that avoids division by zero.

Next, with the data now in a consistent time-series structure, we can consider using a sequence model. However, standard RNNs might still struggle to capture relationships across large time gaps between our newly created time-windows if those gaps were very long in the original series. Therefore, we can consider using an architecture like a Transformer. Transformers, which rely on self-attention mechanisms, can capture dependencies across the entire input sequence, regardless of temporal distance, thus potentially modeling the non-continuous nature of our data more effectively.

The following example demonstrates a basic Transformer model implemented using TensorFlow.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Embedding
from tensorflow.keras.models import Model
import numpy as np

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

def create_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_encoder_layers, output_dim, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    x = inputs #no explicit embedding here, because the input features are already our "embedded" representation
    for _ in range(num_encoder_layers):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Model Parameters
input_shape = (None, 6)  # (time_steps, num_features - 'mean', 'median', 'std', 'min', 'max', 'coverage')
embed_dim = 16
num_heads = 2
ff_dim = 32
num_encoder_layers = 2
output_dim = 1  # single output prediction
dropout_rate = 0.2

# Create and compile the model
model = create_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_encoder_layers, output_dim, dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Dummy Training Data
X_train = np.random.rand(100, 10, 6).astype(np.float32) #batch, time-steps, features
y_train = np.random.rand(100, 1).astype(np.float32) #batch, output dimension

# Training
model.fit(X_train, y_train, epochs=10, verbose = 0)
print("Model training completed.")
```

This code defines a basic Transformer encoder and uses it to build a complete model. The input consists of sequences of statistical features generated from our `create_windowed_features`. Unlike text inputs, there is no explicit position embedding as we assume that position is already encapsulated in the structure of the sequential input. The final `GlobalAveragePooling1D` aggregates the output from the encoder across the time dimension, producing a single vector that is then passed to a dense output layer. The dummy data and training provide an example of how the data is ingested. Note: The `input_shape` assumes that we have six features based on the output of our previous data preprocessing step.

Finally, let's consider a case where the discontinuity in our non-continuous data is very high such that creating windowed features, as in the first code example, would simply discard too much data, since the vast majority of the windows will be empty (or nearly empty) of observations. In such cases, we might consider using a technique known as event-based modeling. Here, we focus not on sampling at regular intervals, but rather at the timestamps where events of interest occur. This may require a hybrid approach, combining the temporal embeddings with the features of each individual event.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, concatenate
from tensorflow.keras.models import Model
import numpy as np

def create_event_model(num_events, num_features, embedding_dim, hidden_units, output_dim):
    """
    Creates a model for event-based time series prediction using embeddings.

    Args:
        num_events (int): Number of unique events.
        num_features (int): Number of non-time series features per event.
        embedding_dim (int): Dimensionality of event embeddings.
        hidden_units (int): Number of units in the hidden layer.
        output_dim (int): Dimensionality of the output.

    Returns:
        tf.keras.Model: Compiled TensorFlow model.
    """
    event_input = Input(shape=(1,), name='event_input', dtype=tf.int32) #integer representing each event type
    feature_input = Input(shape=(num_features,), name='feature_input', dtype=tf.float32)
    time_input = Input(shape=(1,), name='time_input', dtype=tf.float32) #input time, as opposed to a time window

    event_embedding = Embedding(input_dim=num_events, output_dim=embedding_dim)(event_input)
    event_embedding = tf.keras.layers.Flatten()(event_embedding) #flatten the embedding output

    # Time embedding layer: A simple example. Can be more complex.
    time_embedding = Dense(embedding_dim, activation = 'relu')(time_input)

    merged = concatenate([event_embedding, feature_input, time_embedding])
    hidden = Dense(hidden_units, activation='relu')(merged)
    output = Dense(output_dim)(hidden)
    
    model = Model(inputs=[event_input, feature_input, time_input], outputs=output)
    return model

# Model Parameters
num_events = 10 # Assume 10 different event types
num_features = 3 # Assume 3 features per event
embedding_dim = 8
hidden_units = 16
output_dim = 1

# Create and compile model
model = create_event_model(num_events, num_features, embedding_dim, hidden_units, output_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Dummy Training Data
num_samples = 100
events_train = np.random.randint(0, num_events, size=(num_samples, 1)) #integer event types
features_train = np.random.rand(num_samples, num_features).astype(np.float32) #non-time-series feature data per event
times_train = np.random.rand(num_samples, 1).astype(np.float32) #event time
output_train = np.random.rand(num_samples, 1).astype(np.float32) #target value

# Training
model.fit([events_train, features_train, times_train], output_train, epochs=10, verbose = 0)
print("Event-based Model training completed.")

```

This `create_event_model` is structured to take separate inputs for event type (represented as an integer and then embedded), non-time series features associated with each event, and event time (also embedded using a linear transform in this example). The model then combines these via concatenation, which is then followed by dense layers for further processing and prediction. This example demonstrates one particular approach, and it’s possible to use other, more advanced, methods for combining event and time information.

In summary, predicting from non-continuous time series data requires careful preprocessing, feature engineering, and judicious selection of neural network architectures. Windowing-based feature aggregation converts the data into a continuous time-series representation amenable to sequential models. Transformers can capture long-range temporal dependencies, addressing the issue of data gaps in our time series. Event-based models offer an alternative approach where observations are tied to specific events rather than fixed time intervals, allowing us to model sparse data. The specific choice depends on the nature of the data and the specific modeling task. These are strategies I have found valuable in practical applications.

For further learning, I would recommend exploring literature on time series analysis, particularly on data preprocessing techniques for non-uniform sampled data. Textbooks covering deep learning and sequence models, as well as materials focusing on Transformer networks, can prove beneficial. Additionally, research papers dealing with event-based time series analysis can offer insights into more specialized modeling techniques. Practical experience through small projects is invaluable, so working through small-scale implementations is key for understanding the nuances of the techniques.
