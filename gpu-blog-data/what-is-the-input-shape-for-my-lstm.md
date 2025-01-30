---
title: "What is the input shape for my LSTM?"
date: "2025-01-30"
id: "what-is-the-input-shape-for-my-lstm"
---
The critical determinant of the input shape for your LSTM is the nature of your sequential data and the specific LSTM architecture you've chosen.  Over the years, working on various time-series prediction and natural language processing projects, I've found that a fundamental misunderstanding of this aspect leads to frequent errors and inefficient model performance.  The input shape isn't a single, universally applicable value; it's intimately tied to your data's dimensionality and the desired sequence length.

**1. Understanding the LSTM Input Shape:**

An LSTM layer expects input in a three-dimensional tensor.  This tensor represents a batch of sequences, where each sequence is composed of a series of time steps, and each time step contains a feature vector. Therefore, the shape is typically expressed as (samples, timesteps, features).

* **samples:** This dimension represents the number of independent sequences in your batch. During training, this is the batch size.  For example, if you're processing sentences, each sentence is a sample.  If you're analyzing sensor readings, each sensor's readings over a period constitute a sample.

* **timesteps:** This dimension defines the length of each sequence.  In a text processing context, this is the number of words in a sentence. In a sensor data scenario, it could represent the number of data points collected per sensor over a specified time interval.  Variable-length sequences require careful padding or truncation.

* **features:** This dimension describes the dimensionality of the feature vector at each time step.  For textual data, this could be the embedding vector dimension (e.g., word embeddings from Word2Vec or GloVe).  For sensor data, this would be the number of sensor readings per time step.  If you have only a single reading per time step, this dimension will simply be 1.

**2. Code Examples Illustrating Input Shape:**

Let's consider three scenarios and their corresponding code implementations using Keras, a popular deep learning framework I've extensively used.  The examples assume you're using TensorFlow as the backend.

**Example 1: Simple Sentiment Analysis**

This example demonstrates text classification where each word is a feature.  We assume pre-trained word embeddings are used.

```python
import numpy as np
from tensorflow import keras

# Sample data: 5 sentences, each with a maximum of 10 words, using 50-dimensional word embeddings
data = np.random.rand(5, 10, 50)  # (samples, timesteps, features)
labels = np.random.randint(0, 2, 5) # 0 or 1 for positive/negative sentiment

model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(10, 50)), # input_shape explicitly defined
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=10)
```

Here, `input_shape=(10, 50)` explicitly sets the expected input dimensions.  The data is already prepared with padding to ensure all sentences are of length 10.


**Example 2: Multivariate Time Series Forecasting**

This example focuses on forecasting multiple variables over time.

```python
import numpy as np
from tensorflow import keras

# Sample data: 100 time series, each with 50 time steps and 3 features (e.g., temperature, humidity, pressure)
data = np.random.rand(100, 50, 3) # (samples, timesteps, features)
labels = np.random.rand(100, 1) # Predicting a single value

model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences=True, input_shape=(50, 3)), # return_sequences for stacked LSTMs
    keras.layers.LSTM(64),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(data, labels, epochs=10)

```
This demonstrates a stacked LSTM. `return_sequences=True` in the first LSTM layer is crucial for passing the output sequence to the subsequent layer.  The input shape is (50, 3), reflecting the 50 timesteps and 3 features.


**Example 3:  Handling Variable-Length Sequences**

This example shows how to handle sequences of varying lengths using padding.

```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# Sample data: sequences with varying lengths, using a single feature
sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
labels = [0, 1, 0]
max_len = max(len(seq) for seq in sequences)

padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post') # padding at the end
padded_sequences = np.expand_dims(padded_sequences, axis=2) #Add feature dimension for LSTM


model = keras.Sequential([
    keras.layers.LSTM(32, input_shape=(max_len, 1)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10)

```

This illustrates padding using `pad_sequences` from Keras.  The `input_shape` reflects the maximum sequence length and the single feature dimension. Note the explicit addition of the feature dimension using `np.expand_dims`.

**3. Resource Recommendations:**

I strongly recommend consulting the official documentation for Keras and TensorFlow.  Furthermore, exploring textbooks on deep learning, specifically those covering recurrent neural networks, is invaluable.  A good understanding of linear algebra and probability theory will also greatly benefit your comprehension.  Finally, thorough review of code examples from reputable sources, coupled with diligent experimentation, will cement your understanding.
