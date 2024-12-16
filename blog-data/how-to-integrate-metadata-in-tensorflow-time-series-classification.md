---
title: "How to integrate metadata in TensorFlow time series classification?"
date: "2024-12-16"
id: "how-to-integrate-metadata-in-tensorflow-time-series-classification"
---

Alright, let's delve into integrating metadata into TensorFlow time series classification. This is a topic I've tackled multiple times across different projects, and it's always a nuanced challenge. It's not just about concatenating data; it's about intelligently using metadata to improve the model's understanding of the underlying temporal patterns. In my past experience, handling sensor data from industrial machinery proved particularly illuminating on this front. We had readings on vibrations, temperature, and acoustic emissions, all timestamped, alongside contextual information like the machine's operational mode (idle, running, maintenance). Simply feeding raw time series to an LSTM wasn't cutting it. The metadata was essential to understand why certain anomalies appeared.

Firstly, we need to define clearly what constitutes "metadata" in this context. It's any information *not* directly part of the time series that could influence the temporal patterns or the interpretation of said patterns. This might include categorical variables, numerical features, or even text descriptions. For example, in a financial context, this could be company news during specific timestamps, or the current interest rates when the stock prices were recorded. The core challenge is to weave this extra information into a neural network’s processing flow effectively. Simply appending it to the time series vector at each time step is rarely optimal.

I've explored various approaches. A common technique, and one I've found particularly versatile, involves creating separate embeddings for categorical metadata, and possibly normalized numerical features, and then fusing those embeddings with the time-series data at an appropriate stage in the model. Let's illustrate this with a basic example. Assume we have a time series of sensor readings and the machine's operational state as metadata. The operational state has three values: "idle," "running," and "maintenance."

Here is a working code example using TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_metadata_aware_model(time_series_length, num_features, num_categories, embedding_dim):

  time_series_input = keras.Input(shape=(time_series_length, num_features), name="time_series")
  metadata_input = keras.Input(shape=(1,), dtype=tf.int32, name="metadata") # input shape 1 for single categorical value

  # Embedding layer for categorical metadata
  embedding_layer = layers.Embedding(input_dim=num_categories, output_dim=embedding_dim)(metadata_input)

  # Flattening the embedding before processing
  embedded_metadata = layers.Flatten()(embedding_layer)
  
  #Expand metadata to match sequence length
  expanded_metadata = layers.RepeatVector(time_series_length)(embedded_metadata)
  
  # Concatenate metadata with each timestep of time series
  concatenated_input = layers.concatenate([time_series_input, expanded_metadata], axis=-1)

  # Time series processing (e.g., LSTM)
  x = layers.LSTM(64, return_sequences=False)(concatenated_input)
  
  # Fully connected layer for classification
  output = layers.Dense(1, activation='sigmoid')(x)

  model = keras.Model(inputs=[time_series_input, metadata_input], outputs=output)
  return model

# Example Usage
time_series_length = 100
num_features = 3 # Example: accelerometer readings
num_categories = 3 # "idle", "running", "maintenance"
embedding_dim = 10

model = build_metadata_aware_model(time_series_length, num_features, num_categories, embedding_dim)
model.summary()

# Dummy data for example
time_series_data = np.random.rand(10, time_series_length, num_features) # 10 samples
metadata_data = np.random.randint(0, num_categories, size=(10, 1)) # random state for 10 samples

# Compile and train (simplified training for demonstration)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([time_series_data, metadata_data], np.random.randint(0, 2, size=(10, 1)), epochs=2) # Random labels to illustrate setup
```

In this snippet, we first create an `Embedding` layer for the categorical metadata. Then, instead of concatenating the metadata directly to the time series, we repeat the embedding across the temporal dimension (using `RepeatVector`). This ensures the metadata information influences the processing of each time step. After concatenating, the combined input is fed into an LSTM layer, and finally, the output is passed through a dense layer for classification.

However, this approach has its limitations. The metadata, in some instances, might be more influential at specific points in the time series. For example, a machine entering maintenance mode is particularly relevant for anomalies occurring afterward. To address this, I’ve utilized attention mechanisms. By treating the metadata as a context vector, we can direct the network’s attention to different parts of the time series, depending on the metadata itself.

Here's a second, slightly more advanced example incorporating an attention mechanism:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_attention_metadata_model(time_series_length, num_features, num_categories, embedding_dim):
    time_series_input = keras.Input(shape=(time_series_length, num_features), name="time_series")
    metadata_input = keras.Input(shape=(1,), dtype=tf.int32, name="metadata")
    
    # Embedding layer for metadata
    metadata_embedding = layers.Embedding(input_dim=num_categories, output_dim=embedding_dim)(metadata_input)
    embedded_metadata = layers.Flatten()(metadata_embedding)
    
    #expand metadata to match sequence length for context
    expanded_metadata = layers.RepeatVector(time_series_length)(embedded_metadata)
    
    # Time series processing
    lstm_out = layers.LSTM(64, return_sequences=True)(time_series_input)
    
    # Attention layer using metadata as context
    attention = layers.Attention()([lstm_out, expanded_metadata])

    # average out the time sequence dimension before passing to dense layer
    avg_out = layers.GlobalAveragePooling1D()(attention)
    
    # Fully connected layer for classification
    output = layers.Dense(1, activation='sigmoid')(avg_out)
    
    model = keras.Model(inputs=[time_series_input, metadata_input], outputs=output)
    return model


# Example usage
time_series_length = 100
num_features = 3
num_categories = 3
embedding_dim = 10

model = build_attention_metadata_model(time_series_length, num_features, num_categories, embedding_dim)
model.summary()

# Dummy data for example
time_series_data = np.random.rand(10, time_series_length, num_features)
metadata_data = np.random.randint(0, num_categories, size=(10, 1))

# Compile and train (simplified training)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([time_series_data, metadata_data], np.random.randint(0, 2, size=(10, 1)), epochs=2) # Random labels to illustrate setup
```
Here, the output of the LSTM layer now interacts with the extended metadata through the `Attention` layer which will give higher weight to certain steps in the time series.

A third technique, more appropriate if some of the metadata is numerical and potentially very influential, involves early fusion via a multi-modal encoder where each type of data (time series and metadata) has its own processing path, and these paths are only merged at some point. It allows more flexible learning of representations suitable for each input type, rather than concatenating them early on.
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_multimodal_model(time_series_length, num_features, num_categories, num_numerical, embedding_dim):
    time_series_input = keras.Input(shape=(time_series_length, num_features), name="time_series")
    categorical_input = keras.Input(shape=(1,), dtype=tf.int32, name="categorical_metadata")
    numerical_input = keras.Input(shape=(num_numerical,), name="numerical_metadata")

    # Time series encoder
    lstm_out = layers.LSTM(64)(time_series_input) # no return sequence since we use just final state

    # Categorical metadata encoder
    cat_embedding = layers.Embedding(input_dim=num_categories, output_dim=embedding_dim)(categorical_input)
    embedded_cat = layers.Flatten()(cat_embedding)

    # Numerical metadata encoder (simple dense)
    dense_num = layers.Dense(32)(numerical_input)

    # Merging encoders
    merged = layers.concatenate([lstm_out, embedded_cat, dense_num])


    # Fully connected layer for classification
    output = layers.Dense(1, activation='sigmoid')(merged)

    model = keras.Model(inputs=[time_series_input, categorical_input, numerical_input], outputs=output)
    return model


# Example usage
time_series_length = 100
num_features = 3
num_categories = 3
num_numerical = 2
embedding_dim = 10

model = build_multimodal_model(time_series_length, num_features, num_categories, num_numerical, embedding_dim)
model.summary()

# Dummy data for example
time_series_data = np.random.rand(10, time_series_length, num_features)
categorical_data = np.random.randint(0, num_categories, size=(10, 1))
numerical_data = np.random.rand(10, num_numerical)

# Compile and train (simplified)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([time_series_data, categorical_data, numerical_data], np.random.randint(0, 2, size=(10, 1)), epochs=2) # Random labels to illustrate setup
```
This model has separate encoders for each type of data, which may help in certain situations.

Key considerations when implementing these techniques:

1.  **Data Normalization:**  Always normalize numerical metadata and time series data to ensure features contribute equally during learning. I've found batch normalization quite effective, especially when dealing with noisy data.

2.  **Embedding Size:** The embedding dimension for categorical data is a hyperparameter that should be tuned. A larger embedding might capture more complex relationships but can also lead to overfitting if you do not have enough data points per category.

3.  **Temporal Resolution:** If your metadata has a different temporal resolution than your time series, consider time alignment/interpolation. This was a challenge with our sensor data where some contextual info was sampled less frequently than sensor readings.

4.  **Model complexity:** Start with a simple integration of metadata and then increase the complexity as needed. The increased complexity can be very computationally expensive if your dataset is large.

5. **Validation:** Use a proper validation scheme to test for overfitting. This step is crucial when adding additional data sources to your training process.

For further learning, I would recommend looking into the following resources. The first is the book "Deep Learning with Python" by François Chollet. It has a excellent and detailed description of the core components of most neural networks and its use with Keras, which is the building block used in the code examples. A second source is the paper "Attention is All You Need" published by Vaswani et al. in 2017. It is a groundbreaking paper that introduces the attention mechanism and shows how to implement it in various cases. Finally, "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron is a comprehensive guide on practical implementation details of models with TensorFlow. These are resources I've personally consulted and highly recommend.

Integrating metadata into your time series classification isn't a magic bullet, but when done correctly, it allows your model to see the bigger picture, leading to more accurate and reliable predictions. The key is to understand your data, be mindful of the different characteristics of your metadata, and choose the integration technique that best fits those characteristics. Experimentation and careful evaluation are paramount.
