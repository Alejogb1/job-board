---
title: "How can TensorFlow data be shaped for LSTMs in an encoder-decoder model?"
date: "2025-01-30"
id: "how-can-tensorflow-data-be-shaped-for-lstms"
---
Recurrent neural networks, particularly LSTMs, expect input sequences with a specific three-dimensional shape: `[batch_size, time_steps, features]`. In the context of an encoder-decoder architecture, handling diverse input data and preparing it for LSTMs requires careful consideration of batching, sequence length variations, and feature dimensionality. I've encountered numerous challenges in adapting various dataset structures to this format during my time working on sequence-to-sequence models for time-series forecasting, and refining my data-handling procedures has been critical for achieving good performance.

Let's break down the shaping process into distinct stages. First, the raw input data, which could originate from files, databases, or live streams, often exists in a format inconsistent with the LSTM's input requirements. For instance, a time-series dataset might be provided as a single, long sequence of values paired with a corresponding output sequence, or perhaps as a list of varying length sequences. The first step therefore involves data preparation. This typically includes: padding variable-length sequences, normalizing features, creating fixed-length input-output pairs, and then structuring them into training batches.

The core difficulty lies in transforming these diverse input data structures into batches of sequences with uniform length. Padding, a common technique, achieves this by adding placeholder elements, usually zeros, to the end of shorter sequences to match the length of the longest sequence within a batch. Masking layers can subsequently be employed during model training so the padding doesn't influence calculations. For encoder-decoder models, both the encoder's input sequence and the decoder's target sequence need to be appropriately formatted. The encoder's input could be the past history (or observation window) of a time series, while the decoder’s target could be the future data points we wish to predict.

Consider a scenario where we are training a model to predict the next several steps in a time-series based on the prior 10 time points. The data, `raw_data`, is a list of sequences. Each sequence is a NumPy array with a different number of samples (e.g., raw data is not uniformly sized). Here's an example showing how we transform this into batches using a TensorFlow dataset:

```python
import tensorflow as tf
import numpy as np

def create_batches(raw_data, input_len, output_len, batch_size):
    """
    Creates batches of data from variable-length sequences.

    Args:
    raw_data: A list of NumPy arrays, each array representing a time series.
    input_len: The fixed length of the input sequences.
    output_len: The fixed length of the output sequences.
    batch_size: The number of sequences per batch.

    Returns:
    A TensorFlow dataset object.
    """
    all_inputs = []
    all_targets = []

    for seq in raw_data:
        if len(seq) >= input_len + output_len:
            for i in range(len(seq) - input_len - output_len + 1):
                input_seq = seq[i : i + input_len]
                output_seq = seq[i + input_len : i + input_len + output_len]
                all_inputs.append(input_seq)
                all_targets.append(output_seq)

    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((all_inputs, all_targets))
    dataset = dataset.batch(batch_size)
    return dataset

# Example usage with dummy data.
raw_data = [
  np.random.rand(15, 1),
  np.random.rand(20, 1),
  np.random.rand(25, 1)
]

input_length = 10
output_length = 5
batch_size = 2

dataset = create_batches(raw_data, input_length, output_length, batch_size)

for inputs, targets in dataset:
  print("Input shape:", inputs.shape)
  print("Target shape:", targets.shape)
  break # Print just the first batch shape
```

In this example, the `create_batches` function iterates over the list of variable-length sequences. For each sufficiently long sequence, it generates multiple `(input, target)` pairs by sliding a window of size `input_len + output_len`. These pairs are then stored in lists, converted to NumPy arrays, and finally transformed into a TensorFlow dataset, batched to the `batch_size` specified. The output for this example shows the shape `(2, 10, 1)` for inputs, corresponding to two sequences of 10 time steps each, and `(2, 5, 1)` for the corresponding output targets. The third dimension is the number of features (in this case 1). Note, not all the sequences are of equal length, so sequences shorter than `input_length + output_length` are simply ignored.

Now let’s examine how padding can be applied for cases where you may want to retain all data points.

```python
import tensorflow as tf
import numpy as np

def create_padded_batches(raw_data, input_len, output_len, batch_size):
    """
    Creates batches of data from variable-length sequences with padding.

    Args:
      raw_data: A list of NumPy arrays, each array representing a time series.
      input_len: The fixed length of the input sequences.
      output_len: The fixed length of the output sequences.
      batch_size: The number of sequences per batch.

    Returns:
      A TensorFlow dataset object.
    """
    all_inputs = []
    all_targets = []
    max_len = 0

    for seq in raw_data:
        max_len = max(max_len, len(seq))

    for seq in raw_data:
        pad_len = max_len - len(seq)
        padded_seq = np.pad(seq, ((0,pad_len), (0,0)), 'constant')

        for i in range(len(padded_seq) - input_len - output_len + 1):
            input_seq = padded_seq[i: i + input_len]
            output_seq = padded_seq[i + input_len: i + input_len + output_len]
            all_inputs.append(input_seq)
            all_targets.append(output_seq)


    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)


    dataset = tf.data.Dataset.from_tensor_slices((all_inputs, all_targets))
    dataset = dataset.batch(batch_size)
    return dataset

# Example usage with dummy data.
raw_data = [
  np.random.rand(15, 1),
  np.random.rand(20, 1),
  np.random.rand(25, 1)
]

input_length = 10
output_length = 5
batch_size = 2

dataset = create_padded_batches(raw_data, input_length, output_length, batch_size)

for inputs, targets in dataset:
  print("Padded Input shape:", inputs.shape)
  print("Padded Target shape:", targets.shape)
  break # Print just the first batch shape

```

Here, instead of filtering out sequences shorter than `input_len + output_len`, each sequence is padded to the length of the longest sequence before creating input/output pairs. The padding is implemented using `np.pad` to prepend zeros to sequences. The shape of the input will then vary according to the number of data sequences and their corresponding lengths.

Finally, consider a more involved case where we have multiple features:

```python
import tensorflow as tf
import numpy as np

def create_multifeature_batches(raw_data, input_len, output_len, batch_size, num_features):
    """
    Creates batches of data from variable-length multi-feature sequences.

    Args:
      raw_data: A list of NumPy arrays, each array representing a time series. Each item is (sequence_length, num_features).
      input_len: The fixed length of the input sequences.
      output_len: The fixed length of the output sequences.
      batch_size: The number of sequences per batch.
      num_features: The number of features in the data

    Returns:
      A TensorFlow dataset object.
    """

    all_inputs = []
    all_targets = []
    for seq in raw_data:
      if len(seq) >= input_len + output_len:
          for i in range(len(seq) - input_len - output_len + 1):
            input_seq = seq[i : i + input_len]
            output_seq = seq[i + input_len : i + input_len + output_len]
            all_inputs.append(input_seq)
            all_targets.append(output_seq)


    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)


    dataset = tf.data.Dataset.from_tensor_slices((all_inputs, all_targets))
    dataset = dataset.batch(batch_size)
    return dataset

# Example usage with dummy data.
num_features = 3
raw_data = [
  np.random.rand(15, num_features),
  np.random.rand(20, num_features),
  np.random.rand(25, num_features)
]

input_length = 10
output_length = 5
batch_size = 2

dataset = create_multifeature_batches(raw_data, input_length, output_length, batch_size, num_features)

for inputs, targets in dataset:
    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)
    break  # Print just the first batch shape
```
This final code illustrates the same logic for batch creation as in the initial example but with data that contains multiple features. In this instance, the time series data is of shape `(sequence_length, num_features)`. As before, the function creates `(input, target)` pairs of fixed length and stacks them into a TensorFlow dataset. The main difference is the extra dimension for the multiple features. Here, for example, the output is `(2, 10, 3)` corresponding to 2 sequences with 10 time steps and 3 features.

These examples demonstrate fundamental techniques for shaping data for LSTMs. However, depending on data characteristics, additional steps may be required. For example, it may be necessary to normalize features using batch normalization or to create more complex windowing schemes when data is overlapping. Furthermore, one should always pay attention to the number of parameters of the model, as these tend to rapidly increase when the number of features grows.

For more in-depth resources, I suggest consulting textbooks or academic articles on deep learning and time series analysis. Specifically, focus on sections pertaining to sequence-to-sequence modeling and recurrent neural networks. Books from authors such as Goodfellow, Bengio, and Courville will provide a sound theoretical foundation, while publications from conferences like NeurIPS and ICML often cover current best practices. The TensorFlow documentation is also invaluable for staying updated on the most recent API changes.
