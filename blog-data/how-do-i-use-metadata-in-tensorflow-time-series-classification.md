---
title: "How do I use metadata in TensorFlow time series classification?"
date: "2024-12-23"
id: "how-do-i-use-metadata-in-tensorflow-time-series-classification"
---

Right, let’s get into this. It’s not uncommon to encounter time series data where the associated metadata holds significant predictive power, and it’s definitely a case I’ve tackled more than once. I remember a particularly challenging project involving sensor data from industrial machinery; the timestamps alone weren’t enough. We had to incorporate machine type, ambient temperature, and even the technician who last performed maintenance to achieve any semblance of accuracy in predicting failure modes. So, let's unpack how to effectively leverage metadata within a TensorFlow-based time series classification framework.

At its core, the challenge is to fuse time-dependent data with static or slowly changing attributes. TensorFlow provides a flexible environment to handle this, but proper handling is paramount for success. Think about it as having two streams of information that require careful alignment. The time series, typically represented as sequences, and the metadata, which often comes as a set of features for each sequence. The trick is combining them in a way that the model can learn from both.

We typically begin by preprocessing each type of data separately. For time series, this could involve standard techniques like normalization, handling missing values (imputation is preferable to deletion, often), and even feature extraction within each sequence window. Metadata, on the other hand, needs its own handling. Categorical variables, for example, must be one-hot encoded or embedding-mapped into a continuous vector space. Numerical metadata, like the aforementioned ambient temperature, may be normalized or standardized. It's crucial to note that metadata often carries distinct information that can heavily influence the model's performance. It’s not merely an addition; it is often the context within which the time series data operates.

Now, for the model architecture. A common approach involves an encoder for the time series data (a convolutional network or a recurrent network such as lstm or gru are typical starting points), and a separate (or sometimes the same) branch for the metadata. The extracted features from both branches are then concatenated (or combined using more sophisticated techniques) before being fed into subsequent fully connected layers for classification.

Let's illustrate this with a few examples using TensorFlow. Suppose our time series is represented by sequences of shape `(sequence_length, num_features)` and the metadata is represented by a simple vector of shape `(num_metadata_features)`.

**Example 1: Simple Concatenation**

This example showcases a straightforward concatenation approach after both branches process the input. It focuses on clarity rather than state-of-the-art complexity.

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_simple_model(sequence_length, num_features, num_metadata_features, num_classes):
    time_series_input = tf.keras.Input(shape=(sequence_length, num_features))
    metadata_input = tf.keras.Input(shape=(num_metadata_features,))

    # Time Series Encoder (Simple CNN for illustration)
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(time_series_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x) # Flatten to single vector
    time_series_encoded = layers.Dense(128, activation='relu')(x)

    # Metadata Branch
    metadata_encoded = layers.Dense(64, activation='relu')(metadata_input)

    # Concatenate
    merged = layers.concatenate([time_series_encoded, metadata_encoded])

    # Classifier
    output = layers.Dense(num_classes, activation='softmax')(merged)

    model = tf.keras.Model(inputs=[time_series_input, metadata_input], outputs=output)
    return model

# Example usage:
sequence_length = 100
num_features = 3
num_metadata_features = 5
num_classes = 3

model = build_simple_model(sequence_length, num_features, num_metadata_features, num_classes)
model.summary()
```

In this basic model, a simple CNN encodes the time series data, a dense layer processes the metadata, and then the two resulting encoded representations are combined. This model is relatively easy to implement, and often provides a good starting point.

**Example 2: Using Embeddings for Categorical Metadata**

Categorical metadata requires special handling. Embedding layers map the categories into a vector space that is learned during training. This improves representational power. Consider having metadata that includes a categorical feature like 'machine type', with various integers representing each type:

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_embedding_model(sequence_length, num_features, num_metadata_features, num_classes,
                        num_machine_types, embedding_dim):
    time_series_input = tf.keras.Input(shape=(sequence_length, num_features))
    metadata_input = tf.keras.Input(shape=(num_metadata_features,))
    machine_type_input = tf.keras.Input(shape=(1,))  # single integer for machine type

    # Time Series Encoder (LSTM this time)
    x = layers.LSTM(units=64, return_sequences=False)(time_series_input)
    time_series_encoded = layers.Dense(128, activation='relu')(x)

    # Categorical metadata embedding
    machine_type_embedded = layers.Embedding(input_dim=num_machine_types, output_dim=embedding_dim)(machine_type_input)
    machine_type_embedded = layers.Flatten()(machine_type_embedded) # flatten from sequence
    
    # Numerical metadata branch
    metadata_encoded = layers.Dense(64, activation='relu')(metadata_input)
    
    # Combine metadata features
    merged_metadata = layers.concatenate([machine_type_embedded, metadata_encoded])

    # Concatenate all inputs
    merged = layers.concatenate([time_series_encoded, merged_metadata])
    
    # Classifier
    output = layers.Dense(num_classes, activation='softmax')(merged)

    model = tf.keras.Model(inputs=[time_series_input, metadata_input, machine_type_input], outputs=output)
    return model

# Example usage:
sequence_length = 100
num_features = 3
num_metadata_features = 4
num_classes = 3
num_machine_types = 5
embedding_dim = 16

model = build_embedding_model(sequence_length, num_features, num_metadata_features, num_classes,
                            num_machine_types, embedding_dim)
model.summary()
```

Here, we've included an `Embedding` layer to handle the categorical 'machine type' feature. This layer learns a distributed representation for each type. The output from the embedding layer is then combined with other numerical metadata prior to concatenation with the time series encoding.

**Example 3: Advanced Fusion Techniques**

More advanced methods include attention mechanisms that can dynamically determine which parts of the time series are most relevant given the metadata, or even specialized recurrent networks that ingest both data types sequentially. While demonstrating these is more involved, I will present a skeleton concept for context-aware fusion.

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_attention_model(sequence_length, num_features, num_metadata_features, num_classes):
    time_series_input = tf.keras.Input(shape=(sequence_length, num_features))
    metadata_input = tf.keras.Input(shape=(num_metadata_features,))
    
    # Time Series Encoder
    lstm = layers.LSTM(units=64, return_sequences=True)(time_series_input)

    # Metadata Branch
    metadata_encoded = layers.Dense(64, activation='relu')(metadata_input)

    # Attention mechanism (simplified)
    attention_weights = layers.Dense(sequence_length, activation='softmax')(metadata_encoded) # Weights based on metadata
    context_vector = layers.Dot(axes=(1,1))([attention_weights, tf.transpose(lstm, perm=[0,2,1])]) # weighted time series feature
    context_vector = layers.Permute((2,1))(context_vector) # re-permute to have consistent time series dimension

    # Combine with original LSTM output (optional, depends on desired functionality)
    merged = layers.concatenate([lstm, context_vector])
    merged = layers.GlobalAveragePooling1D()(merged)
    
    # Classifier
    output = layers.Dense(num_classes, activation='softmax')(merged)
    
    model = tf.keras.Model(inputs=[time_series_input, metadata_input], outputs=output)
    return model

# Example usage
sequence_length = 100
num_features = 3
num_metadata_features = 5
num_classes = 3

model = build_attention_model(sequence_length, num_features, num_metadata_features, num_classes)
model.summary()
```

This snippet demonstrates a very basic form of attention where metadata influences how much attention is paid to specific steps in the sequence. In practice, attention would often be far more nuanced. It’s crucial to note that this is a more involved approach and could be the direction to explore after experimenting with the prior ones.

For further learning, I would highly recommend exploring the following: "Deep Learning with Python" by François Chollet for the fundamentals of keras and deep learning. For specific time-series related architectures, "Hands-On Time Series Analysis with Python" by Ava Sharma should be useful. Papers such as "Attention is All You Need" by Vaswani et al, which introduced the transformer architecture, can serve as a foundation for understanding more advanced approaches that can leverage metadata in complex ways.

The key to success with metadata in time series classification is understanding the nature of your data, preprocessing it thoughtfully, choosing an appropriate architecture for fusion, and of course, experimentation. Start simple, iterate, and always evaluate based on metrics relevant to your specific problem.
