---
title: "How can I input two ragged tensors into a TensorFlow model?"
date: "2025-01-26"
id: "how-can-i-input-two-ragged-tensors-into-a-tensorflow-model"
---

Ragged tensors, by their very nature of having varying lengths along one or more dimensions, present unique challenges when used as inputs to TensorFlow models designed for uniform tensor shapes. The typical approach of directly passing two ragged tensors into a `tf.keras.Model` without proper pre-processing will lead to shape mismatches during the model’s computation. I’ve encountered this exact scenario numerous times when developing sequence models for natural language processing, and the solution involves understanding how to pad and potentially mask these variable-length sequences prior to model ingestion.

The core issue lies in the expectation that most TensorFlow layers, particularly dense, convolutional, and recurrent layers, require inputs with fixed shapes. Ragged tensors, while incredibly useful for representing data with inconsistent lengths, violate this expectation. Consequently, a preprocessing step is essential to reconcile the variable shape of ragged tensors with the fixed-shape input requirements of the model. We accomplish this through padding, converting each ragged tensor to a dense tensor with the same shape across each batch. Depending on the downstream model, the padding may need to be masked to prevent padding tokens from interfering with the prediction.

Consider a scenario where we have two ragged tensors, `ragged_tensor_a` and `ragged_tensor_b`, each representing sequences of varying lengths. I frequently have to handle this when dealing with user behavior logs, where each user interacts with a varying number of items. Let’s say we want to pass these sequences to a model that performs some kind of feature extraction on each sequence and then combines the results. We need to transform these ragged tensors into dense tensors first, typically by padding them to the same length.

The primary tool for achieving this is `tf.keras.preprocessing.sequence.pad_sequences` when dealing with integer encoded sequences (or other numeric representations) or `tf.ragged.to_tensor` which provides fine grain control on the padding value and shape for arbitrary tensors. The latter method offers more flexibility and control, especially for situations beyond simple sequence padding. Let’s consider these with code examples.

**Example 1: Basic Padding and Masking with `tf.ragged.to_tensor`**

This example illustrates how we can transform two ragged tensors into dense tensors and create a corresponding mask tensor to avoid padding effects.

```python
import tensorflow as tf

ragged_tensor_a = tf.ragged.constant([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
ragged_tensor_b = tf.ragged.constant([[10, 11], [12], [13, 14, 15]])

# Convert to dense tensors with padding, using 0 as the padding value.
padded_a = ragged_tensor_a.to_tensor(default_value=0)
padded_b = ragged_tensor_b.to_tensor(default_value=0)

# Create a mask to identify valid (non-padded) elements
mask_a = tf.cast(ragged_tensor_a.to_tensor(default_value=0) != 0, dtype=tf.float32)
mask_b = tf.cast(ragged_tensor_b.to_tensor(default_value=0) != 0, dtype=tf.float32)

print("Padded Tensor A:\n", padded_a)
print("Padded Tensor B:\n", padded_b)
print("Mask Tensor A:\n", mask_a)
print("Mask Tensor B:\n", mask_b)

# Example of the padded input to a keras model, utilizing the masks
input_a = tf.keras.layers.Input(shape=padded_a.shape[1:], dtype=tf.int32)
input_b = tf.keras.layers.Input(shape=padded_b.shape[1:], dtype=tf.int32)
mask_input_a = tf.keras.layers.Input(shape=mask_a.shape[1:], dtype=tf.float32)
mask_input_b = tf.keras.layers.Input(shape=mask_b.shape[1:], dtype=tf.float32)


embedding_layer_a = tf.keras.layers.Embedding(input_dim=10, output_dim=64, mask_zero=True)(input_a)
embedding_layer_b = tf.keras.layers.Embedding(input_dim=16, output_dim=64, mask_zero=True)(input_b)

# Option 1: masking on model layer level
lstm_layer_a = tf.keras.layers.LSTM(32)(embedding_layer_a, mask=mask_input_a)
lstm_layer_b = tf.keras.layers.LSTM(32)(embedding_layer_b, mask=mask_input_b)


# Option 2: masking via tf.math.multiply:
lstm_layer_a_unmasked = tf.keras.layers.LSTM(32)(embedding_layer_a)
lstm_layer_b_unmasked = tf.keras.layers.LSTM(32)(embedding_layer_b)
lstm_layer_a_masked = tf.math.multiply(lstm_layer_a_unmasked, tf.expand_dims(mask_input_a,-1))
lstm_layer_b_masked = tf.math.multiply(lstm_layer_b_unmasked, tf.expand_dims(mask_input_b,-1))


concatenated = tf.keras.layers.concatenate([lstm_layer_a, lstm_layer_b])
# concattenated_masked = tf.keras.layers.concatenate([lstm_layer_a_masked, lstm_layer_b_masked])

output = tf.keras.layers.Dense(1)(concatenated)

model = tf.keras.Model(inputs=[input_a,input_b,mask_input_a,mask_input_b], outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
model.summary()

# We can now pass the padded tensors and their masks
model.fit([padded_a,padded_b,mask_a,mask_b], tf.random.uniform((padded_a.shape[0], 1), maxval = 2, dtype = tf.int32), epochs = 3)
```

In this first example, we utilize `to_tensor` to explicitly convert each ragged tensor into dense tensors, using 0 as the padding value. Crucially, we create boolean masks that correspond to the padded tensors, thereby indicating which values are valid data points and which are introduced padding. These masks are essential when we employ masking layers like `Embedding` with `mask_zero=True`, which ensures the padding values are ignored during calculations. I've seen substantial performance improvements by properly utilizing masks in models. Note that masking is not mandatory, it depends on the downstream model if masking is required or not.

**Example 2: Padding with Different Maximum Lengths**

In certain situations, each ragged tensor might have different length constraints. `to_tensor` offers a parameter `shape` that allows us to enforce an upper bound on their lengths.

```python
import tensorflow as tf
ragged_tensor_a = tf.ragged.constant([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
ragged_tensor_b = tf.ragged.constant([[10, 11], [12], [13, 14, 15,16,17]])
max_length_a = 5
max_length_b = 3

# Pad tensor A to a max length of 5
padded_a = ragged_tensor_a.to_tensor(default_value=0, shape=[None, max_length_a])
# Pad tensor B to max length 3, truncating if needed
padded_b = ragged_tensor_b.to_tensor(default_value=0, shape=[None, max_length_b])

# Create masks
mask_a = tf.cast(padded_a != 0, dtype=tf.float32)
mask_b = tf.cast(padded_b != 0, dtype=tf.float32)

print("Padded Tensor A:\n", padded_a)
print("Padded Tensor B:\n", padded_b)
print("Mask Tensor A:\n", mask_a)
print("Mask Tensor B:\n", mask_b)


input_a = tf.keras.layers.Input(shape=padded_a.shape[1:], dtype=tf.int32)
input_b = tf.keras.layers.Input(shape=padded_b.shape[1:], dtype=tf.int32)
mask_input_a = tf.keras.layers.Input(shape=mask_a.shape[1:], dtype=tf.float32)
mask_input_b = tf.keras.layers.Input(shape=mask_b.shape[1:], dtype=tf.float32)

embedding_layer_a = tf.keras.layers.Embedding(input_dim=10, output_dim=64)(input_a)
embedding_layer_b = tf.keras.layers.Embedding(input_dim=18, output_dim=64)(input_b)

# Option 1: Masking via the model layer:
lstm_layer_a = tf.keras.layers.LSTM(32)(embedding_layer_a, mask=mask_input_a)
lstm_layer_b = tf.keras.layers.LSTM(32)(embedding_layer_b, mask=mask_input_b)


concatenated = tf.keras.layers.concatenate([lstm_layer_a, lstm_layer_b])
output = tf.keras.layers.Dense(1)(concatenated)

model = tf.keras.Model(inputs=[input_a,input_b,mask_input_a,mask_input_b], outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
model.summary()
model.fit([padded_a,padded_b,mask_a,mask_b], tf.random.uniform((padded_a.shape[0], 1), maxval = 2, dtype = tf.int32), epochs = 3)

```

Here, we see the `shape` parameter of `to_tensor` in action. Each tensor is padded to its specified maximum length, and tensor b is even truncated when it exceeds the maximal length which is 3. This can be important when dealing with potentially long sequences and limited computational resources.

**Example 3: Using `tf.keras.preprocessing.sequence.pad_sequences`**

In cases where the ragged tensors are directly derived from sequences of integers, you can use `tf.keras.preprocessing.sequence.pad_sequences` as an alternative.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

ragged_list_a = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
ragged_list_b = [[10, 11], [12], [13, 14, 15]]
max_length_a = 6
max_length_b = 5

padded_a = pad_sequences(ragged_list_a, padding='post', maxlen=max_length_a)
padded_b = pad_sequences(ragged_list_b, padding='post', maxlen=max_length_b)

mask_a = tf.cast(padded_a != 0, dtype=tf.float32)
mask_b = tf.cast(padded_b != 0, dtype=tf.float32)

print("Padded Tensor A:\n", padded_a)
print("Padded Tensor B:\n", padded_b)
print("Mask Tensor A:\n", mask_a)
print("Mask Tensor B:\n", mask_b)

input_a = tf.keras.layers.Input(shape=padded_a.shape[1:], dtype=tf.int32)
input_b = tf.keras.layers.Input(shape=padded_b.shape[1:], dtype=tf.int32)
mask_input_a = tf.keras.layers.Input(shape=mask_a.shape[1:], dtype=tf.float32)
mask_input_b = tf.keras.layers.Input(shape=mask_b.shape[1:], dtype=tf.float32)

embedding_layer_a = tf.keras.layers.Embedding(input_dim=10, output_dim=64)(input_a)
embedding_layer_b = tf.keras.layers.Embedding(input_dim=16, output_dim=64)(input_b)

lstm_layer_a = tf.keras.layers.LSTM(32)(embedding_layer_a, mask=mask_input_a)
lstm_layer_b = tf.keras.layers.LSTM(32)(embedding_layer_b, mask=mask_input_b)


concatenated = tf.keras.layers.concatenate([lstm_layer_a, lstm_layer_b])
output = tf.keras.layers.Dense(1)(concatenated)

model = tf.keras.Model(inputs=[input_a,input_b,mask_input_a,mask_input_b], outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
model.summary()
model.fit([padded_a,padded_b,mask_a,mask_b], tf.random.uniform((padded_a.shape[0], 1), maxval = 2, dtype = tf.int32), epochs = 3)

```
In this last example, `pad_sequences` handles the padding directly using integer list as input. The `padding` parameter specifies where the padding is added ('post' being after the sequence data), and `maxlen` defines the max sequence length. The main difference from `to_tensor` is the input data type, which must be a list of lists where each inner list represents a sequence, typically with integer encodings.

In summary, to successfully input two ragged tensors into a TensorFlow model, we must first pad each tensor to have a fixed length, handling the differences in sequence lengths. The most effective way is using either `tf.ragged.to_tensor` or `tf.keras.preprocessing.sequence.pad_sequences`, depending on your specific needs and data format. Both techniques are critical in my workflow when dealing with sequential data. The use of masking tensors is highly advised, but not mandatory, and prevents padding values from incorrectly influencing the model's outputs, especially when combined with `Embedding` layers and other layers that can recognize masks.

For further information, the official TensorFlow documentation on ragged tensors and sequence preprocessing can provide a more detailed exposition. Additionally, research papers on recurrent neural network architectures and sequence modeling typically discuss these preprocessing steps in the context of specific tasks. The "TensorFlow in Practice" course on Coursera, and other online educational platforms also address these issues with practical examples.
