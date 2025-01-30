---
title: "How do I specify the `steps` argument when using data tensors as model input?"
date: "2025-01-30"
id: "how-do-i-specify-the-steps-argument-when"
---
When training a recurrent neural network (RNN), particularly with sequences of varying lengths, correctly specifying the `steps` argument during input tensor processing becomes essential for achieving accurate and efficient learning. Incorrect handling often leads to masked data influencing calculations or misaligned batch processing, severely impacting model performance. I've encountered these challenges frequently in projects involving time-series analysis and natural language processing, where varying sequence lengths are the norm, not the exception. Understanding how `steps` interacts with data tensor structure and the underlying Keras mechanisms is critical.

At its core, the `steps` argument determines how many time steps, or sequence elements, from a given input tensor are considered in each training iteration. It's specifically relevant when using data that has been appropriately organized into batches of sequences, and when the data sequences may not be of uniform length. Keras' RNN layers, such as `LSTM` and `GRU`, are designed to process such temporal data. However, without proper guidance through `steps`, the model may struggle to learn relationships within the sequences. The input tensors typically have a shape that includes `(batch_size, steps, features)`, or in some instances, have an extra trailing dimension like `(batch_size, steps, features, channels)` when dealing with multi-channel data.

The value provided to `steps` isn't necessarily the total length of each sequence, but rather the size of the time window that is presented to the RNN at a single training step. Think of it as the horizontal width of the mini-sequence that will be passed to the recurrent layer at a time. This granularity of training allows for more efficient handling of longer sequences by dividing them into shorter manageable segments. Importantly, not providing `steps` or providing an incorrect value will cause Keras to infer some default behavior, which is frequently not what you intended and can produce unexpected results. The default behavior is to process the complete sequence as a single step, which is often not viable for very long sequences and is especially troublesome when dealing with varying sequence lengths within a batch.

The underlying mechanics involve Keras iterating through the input tensor, using the `steps` argument as a stride or window size along the time axis. This implies that if sequences are longer than the `steps` value, then multiple iterations are needed to process the complete sequence. The model's state is preserved between the iterations (for each sequence in the batch) allowing the network to maintain context across those sequence subsections. For sequences shorter than `steps`, masking is typically required to ignore padded elements within the sequence. Masking is essential because these padded positions are essentially null data, and the padding needs to be discounted to avoid skewing results.

To illustrate these points, let’s look at several coding examples.

**Example 1: Defining a Simple LSTM Model with a Fixed `steps` Value**

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np

# Example data: batch of 3 sequences, each 10 steps long, 5 features per step
data = np.random.rand(3, 10, 5)

# We will process each sequence in chunks of 5 steps
steps_value = 5
num_features = 5
num_units = 32  # Number of LSTM units

model = Sequential()
model.add(LSTM(num_units, input_shape=(steps_value, num_features), return_sequences=False))
model.add(Dense(1, activation='sigmoid')) # Binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate placeholder labels
labels = np.random.randint(0, 2, size=(3,))

# Reshape data such that it matches the shape anticipated by the model
# data_reshaped = data.reshape(3,-1,5) - note, this is *not* what we need in this scenario!

# Train the model using data split into chunks of specified step length
# NOTE: We do not explicitly pass any 'steps' argument to model.fit, it is inferred from
#       the shapes of the data being input.
model.fit(data, labels, epochs=10, verbose=0)

print("Model trained successfully.")

```

In this example, the input `data` tensor has a shape of `(3, 10, 5)`, representing 3 sequences, each with 10 steps and 5 features. However, in the `LSTM` layer declaration, `input_shape` is set to `(steps_value, num_features)`, or `(5, 5)`. This means the layer is designed to accept sequences of 5 time steps at a time.  The crucial observation is that within the call to `model.fit`, there isn't an explicit reference to `steps`. Instead, it implicitly takes the size of the 'steps' dimension from the *input shape* of the tensor being fed into it.  Keras then intelligently uses the information from the shape of the data tensor itself during training, splitting the overall sequence into time-segments of length `steps_value`, in this case `5`, and preserving the state between these segments as it iterates through the length of sequence within the batch. While I have shown data that already aligns with how the model wants its input (i.e. shape (batch, steps, features)), what if our sequences were *not* of a length divisible by the steps_value? We would need to either pre-process by padding shorter sequences up to a certain limit, or by masking.

**Example 2: Handling Variable-Length Sequences with Masking**

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# Example data: a batch of sequences of varying length
sequences = [
    np.random.rand(7, 3),   # 7 steps, 3 features
    np.random.rand(12, 3),  # 12 steps, 3 features
    np.random.rand(4, 3)    # 4 steps, 3 features
]

# Pad sequences to the maximum length in the batch (12 in this case)
padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')

# Now padded_sequences is of shape (3, 12, 3), where 3 is the batch size,
#  12 is the maximum sequence length, and 3 is the feature dimension.
max_length = 12
steps_value = 5  # Process in 5-step chunks
num_features = 3
num_units = 32

model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(max_length, num_features))) # Specify masking
model.add(LSTM(num_units, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

labels = np.random.randint(0, 2, size=(3,))

# Train the model on the padded sequences.
model.fit(padded_sequences, labels, epochs=10, verbose=0)

print("Model trained with masking successfully.")
```
In the second example, I am processing variable length sequences. Firstly, I used `pad_sequences` to standardize the sequence lengths within a batch. Critically, the masking layer is *before* the LSTM layer. The `Masking` layer uses a `mask_value` parameter to tell Keras which locations are padding and are to be ignored. Without this masking layer, those 0-filled locations from `pad_sequences` would be processed as valid values. The `LSTM` layer in this example, just like in example 1, doesn't receive the `steps` argument explicitly; instead it implicitly derives this from the `input_shape` and processes the sequence in chunks of length `steps_value = 5`.  This showcases how to effectively use masking combined with the understanding of `steps` to properly deal with variable length sequences. The mask propagates during training within the Keras framework.

**Example 3: Processing a Full Sequence Length in One Step**

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np

# Example data: batch of 3 sequences, of varying lengths, with 3 features per step
sequences = [
    np.random.rand(7, 3),  # 7 steps, 3 features
    np.random.rand(12, 3),  # 12 steps, 3 features
    np.random.rand(4, 3)   # 4 steps, 3 features
]


# Pad sequences to the maximum length in the batch
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', dtype='float32')
max_length = padded_sequences.shape[1]
num_features = 3
num_units = 32


model = Sequential()
model.add(LSTM(num_units, input_shape=(max_length, num_features), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

labels = np.random.randint(0, 2, size=(3,))

model.fit(padded_sequences, labels, epochs=10, verbose=0)

print("Model trained with full sequence processing.")
```

Here, the `steps` value is implied as equal to the `max_length` (i.e. the length of the padded sequence) because that is what's specified in `input_shape`. While the code here doesn't explicitly use the `steps` argument in `LSTM`, it does illustrate a slightly different behavior where the entire sequence in each batch is processed as one single chunk. This approach may be reasonable for certain problems, especially with relatively short sequences. But for longer sequences, this can be computationally intensive. This final example is intended to help contrast with the first two, where it's more common to set `steps` to a fixed value that is significantly less than the total length of the padded sequence, if not already handled during data ingestion. In other words, the implicit value of `steps` is important.

For further information on managing sequence data with Keras, I recommend exploring the documentation related to RNN layers (`LSTM`, `GRU`), the `Masking` layer, and the sequence preprocessing utilities including `pad_sequences`. Reading tutorials and guides related to time series analysis or natural language processing will be invaluable for contextual understanding of these elements. Also, reviewing code examples related to relevant tasks can be a helpful source of practical insight into using Keras in real-world scenarios. These various resources collectively can help clarify the nuances of temporal data and how to use Keras to properly handle it.
