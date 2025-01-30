---
title: "How can I embed float features into a DCN model?"
date: "2025-01-30"
id: "how-can-i-embed-float-features-into-a"
---
Embedding float features into a Deep & Cross Network (DCN) requires careful consideration of how these continuous values interact within the model's architecture, particularly given DCN's strength in capturing feature interactions. Unlike categorical features, which are naturally handled via embeddings, floating-point values need a different approach. Simply attempting to treat them as indices into an embedding matrix will result in nonsensical results, as their continuous nature implies a potentially infinite space of values and therefore, a potentially infinite embedding matrix.

My experience building recommendation systems has shown that a common and effective technique is to first normalize and then either discretize or directly feed the float features into the model, often alongside embeddings of other (categorical) features. Normalization ensures that the feature values lie within a reasonable range, preventing numerical instability during training. Discretization allows for leveraging embedding layers, whereas direct injection typically utilizes a linear layer or other nonlinear projection. The choice depends heavily on the feature distribution, the desired model complexity, and the available compute resources.

The core problem lies in the fact that DCN is designed to efficiently learn interactions between categorical features through embedding lookups and subsequent cross-product operations. This mechanism breaks down with continuous variables, since there is no discrete set of pre-defined "categories" to associate embeddings with.

**Explanation:**

The standard DCN architecture involves an embedding layer to transform sparse categorical input into dense vectors. These embeddings are then fed into both a deep neural network (DNN) and a cross network (Cross Network). The DNN learns complex nonlinear interactions, whereas the Cross Network explicitly models feature interactions using repeated cross-product operations. The combination enables the model to effectively capture both linear and high-order dependencies.

Float features, on the other hand, represent a continuous spectrum. Direct embedding is not applicable. We therefore must employ data preprocessing techniques prior to integrating them with the categorical embeddings. Normalization, for example using min-max scaling or z-score standardization, ensures that the float features do not overwhelm other inputs during training. These normalized features can then be either directly included in the DNN and Cross network’s input, alongside the embeddings, or discretized into buckets. In the case of discretization, each bucket's boundaries are established based on the range and distribution of the features and each bucket is then considered as a category whose integer representation indexes an embedding matrix, similar to other categorical features. This converts our float input into an intermediary discrete representation. The crucial difference is that a single float feature now contributes an embedding vector, in addition to its original value or discretized bucket index.

Once the float features have been preprocessed, the next step involves feeding these transformed inputs into the DCN's deep and cross networks alongside the categorical embeddings. One strategy involves concatenating all inputs: the float features (either directly after normalization or after embedding), together with the categorical embeddings. This single long input vector then serves as the input to the DNN and Cross layers. The combined model can now learn relationships between categorical and continuous features simultaneously.

**Code Examples:**

Here are three code examples, demonstrating different methods of integrating float features into a DCN model. These examples use TensorFlow but the core concepts are transferable to other deep learning frameworks.

**Example 1: Direct Injection of Normalized Float Features**

This example demonstrates a direct injection of normalized float features after concatenation with embeddings.

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_dcn_with_float_direct(num_categorical_features, embedding_dims, num_float_features, dense_units=[64, 32]):
    categorical_inputs = [layers.Input(shape=(1,), dtype=tf.int32, name=f'cat_{i}')
                           for i in range(num_categorical_features)]
    float_inputs = layers.Input(shape=(num_float_features,), dtype=tf.float32, name='float_features')

    embedding_layers = [layers.Embedding(input_dim=embedding_dims[i], output_dim=embedding_dims[i])
                        for i in range(num_categorical_features)]
    embedded_outputs = [embedding_layers[i](categorical_inputs[i]) for i in range(num_categorical_features)]
    embedded_concat = layers.Concatenate(axis=-1)(embedded_outputs)

    all_inputs = layers.Concatenate(axis=-1)([embedded_concat, float_inputs])


    # Deep Network
    dnn_output = all_inputs
    for units in dense_units:
        dnn_output = layers.Dense(units, activation='relu')(dnn_output)


    # Cross Network
    cross_output = all_inputs
    for _ in range(3):
        cross_output = layers.Dense(all_inputs.shape[-1])(tf.matmul(tf.expand_dims(cross_output,axis=2), tf.expand_dims(all_inputs,axis=1))) + cross_output


    # Output Layer
    concat_output = layers.Concatenate(axis=-1)([dnn_output, cross_output])
    final_output = layers.Dense(1, activation='sigmoid')(concat_output)


    model = tf.keras.Model(inputs=categorical_inputs + [float_inputs], outputs=final_output)
    return model

num_categorical_features = 3
embedding_dims = [100, 50, 25]
num_float_features = 2
model = build_dcn_with_float_direct(num_categorical_features, embedding_dims, num_float_features)
model.summary()
```

*Commentary:* This example directly uses the normalized float features by concatenating them with the embeddings of categorical features. The concatenated inputs are then fed into the DNN and Cross Networks. We avoid the need to discretize the float features, resulting in a potentially more nuanced representation if sufficiently high quality normalization is performed.

**Example 2: Discretization and Embedding of Float Features**

This example demonstrates how to discretize and embed float features.

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def build_dcn_with_float_discretized(num_categorical_features, embedding_dims, num_float_features, num_buckets, dense_units=[64, 32]):

    categorical_inputs = [layers.Input(shape=(1,), dtype=tf.int32, name=f'cat_{i}')
                           for i in range(num_categorical_features)]
    float_inputs = layers.Input(shape=(num_float_features,), dtype=tf.float32, name='float_features')
    bucketized_inputs = []
    float_embedding_dims = [20 for i in range(num_float_features)]

    for i in range(num_float_features):
        # Simulate bucketizing by adding an offset to the input
        # In reality, you would use `tf.bucketize` based on thresholds learned from the data.
        bucketized_feature = layers.Lambda(lambda x: tf.cast(x[:,i]*10,tf.int32))(float_inputs)
        bucketized_inputs.append(layers.Embedding(input_dim=num_buckets,output_dim=float_embedding_dims[i])(bucketized_feature))


    embedding_layers = [layers.Embedding(input_dim=embedding_dims[i], output_dim=embedding_dims[i])
                        for i in range(num_categorical_features)]
    embedded_outputs = [embedding_layers[i](categorical_inputs[i]) for i in range(num_categorical_features)]
    embedded_concat = layers.Concatenate(axis=-1)(embedded_outputs)

    bucket_concat = layers.Concatenate(axis=-1)(bucketized_inputs)

    all_inputs = layers.Concatenate(axis=-1)([embedded_concat, bucket_concat])



    # Deep Network
    dnn_output = all_inputs
    for units in dense_units:
        dnn_output = layers.Dense(units, activation='relu')(dnn_output)


    # Cross Network
    cross_output = all_inputs
    for _ in range(3):
        cross_output = layers.Dense(all_inputs.shape[-1])(tf.matmul(tf.expand_dims(cross_output,axis=2), tf.expand_dims(all_inputs,axis=1))) + cross_output

    # Output Layer
    concat_output = layers.Concatenate(axis=-1)([dnn_output, cross_output])
    final_output = layers.Dense(1, activation='sigmoid')(concat_output)


    model = tf.keras.Model(inputs=categorical_inputs + [float_inputs], outputs=final_output)
    return model

num_categorical_features = 3
embedding_dims = [100, 50, 25]
num_float_features = 2
num_buckets = 50
model = build_dcn_with_float_discretized(num_categorical_features, embedding_dims, num_float_features, num_buckets)
model.summary()
```

*Commentary:* In this approach, each float feature is discretized into a fixed number of buckets, and each bucket is represented by an embedding. This allows the model to capture non-linear relationships between these features and their relation to other categories within the model. The number of buckets should be tuned based on data distribution and model performance. In a real application, `tf.bucketize` should be used with bucket boundaries derived from data analysis.

**Example 3: Concatenating Linear Projections of Float Features with Embeddings**

This method applies linear projections to float features before concatenating with categorical embeddings.

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_dcn_with_float_projection(num_categorical_features, embedding_dims, num_float_features, projection_dim = 20, dense_units=[64, 32]):

    categorical_inputs = [layers.Input(shape=(1,), dtype=tf.int32, name=f'cat_{i}')
                           for i in range(num_categorical_features)]
    float_inputs = layers.Input(shape=(num_float_features,), dtype=tf.float32, name='float_features')


    projected_floats = layers.Dense(projection_dim)(float_inputs)


    embedding_layers = [layers.Embedding(input_dim=embedding_dims[i], output_dim=embedding_dims[i])
                        for i in range(num_categorical_features)]
    embedded_outputs = [embedding_layers[i](categorical_inputs[i]) for i in range(num_categorical_features)]
    embedded_concat = layers.Concatenate(axis=-1)(embedded_outputs)

    all_inputs = layers.Concatenate(axis=-1)([embedded_concat, projected_floats])



    # Deep Network
    dnn_output = all_inputs
    for units in dense_units:
        dnn_output = layers.Dense(units, activation='relu')(dnn_output)


    # Cross Network
    cross_output = all_inputs
    for _ in range(3):
        cross_output = layers.Dense(all_inputs.shape[-1])(tf.matmul(tf.expand_dims(cross_output,axis=2), tf.expand_dims(all_inputs,axis=1))) + cross_output


    # Output Layer
    concat_output = layers.Concatenate(axis=-1)([dnn_output, cross_output])
    final_output = layers.Dense(1, activation='sigmoid')(concat_output)


    model = tf.keras.Model(inputs=categorical_inputs + [float_inputs], outputs=final_output)
    return model


num_categorical_features = 3
embedding_dims = [100, 50, 25]
num_float_features = 2
projection_dim = 20
model = build_dcn_with_float_projection(num_categorical_features, embedding_dims, num_float_features, projection_dim)
model.summary()

```

*Commentary:* This method uses a linear layer to project the float features into a higher-dimensional space before concatenation with embeddings, potentially allowing for the learning of more complex relationships with the categorical inputs. The projection dimension parameter serves as a tunable hyperparameter to balance model complexity and performance. A nonlinear activation can be added to this projection layer.

**Resource Recommendations:**

For gaining a deeper understanding of DCNs, consulting research papers discussing the architecture is valuable. Materials covering feature engineering for machine learning provide a solid foundation for normalizing and discretizing continuous features. Books on deep learning can clarify concepts such as embeddings, neural networks, and cross-product operations. Furthermore, practicing with the TensorFlow/Keras API (or similar libraries), through experimentation with different embedding dimensions and model architectures, is vital for building practical experience. Focus on both architecture understanding and hands-on experimentation to achieve optimal model results.
