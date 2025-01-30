---
title: "How do I convert a list of lists of NumPy arrays to a PyTorch tensor suitable for LSTM?"
date: "2025-01-30"
id: "how-do-i-convert-a-list-of-lists"
---
Here’s how I’ve approached converting nested lists of NumPy arrays into a PyTorch tensor for LSTM input, given that it's a task I've often faced when working with sequential data having varying lengths across different instances. The core challenge lies in PyTorch's expectation of a regular, uniformly sized tensor, while nested lists can represent sequences of variable lengths, or even have different array shapes within. The solution invariably involves padding.

**Understanding the Impediment**

PyTorch's LSTM layers require input tensors of a specific shape, typically `(sequence_length, batch_size, input_size)` when batch-first is `False` or `(batch_size, sequence_length, input_size)` when batch-first is `True`. When the data arrives as a list of lists of NumPy arrays, the sequences within each outer list (representing a batch) can have varying lengths, and the inner lists may contain arrays of different shape. A direct conversion using `torch.tensor` will fail because tensors require uniformity in size across all dimensions. This heterogeneity, common in many time-series or sequence modeling scenarios, necessitates careful preprocessing.

**The Solution: Padding and Tensor Construction**

My approach centers on padding the shorter sequences to match the longest one within the batch and then constructing a PyTorch tensor. The key is to identify the maximum sequence length within a batch, create a zero-filled tensor with dimensions determined by that maximum length and the array shape, and copy the contents of the original data into the appropriate positions of that tensor. Below, I provide a Python function and examples that illustrate this approach.

**Code Example 1: Basic Padding and Conversion**

This snippet demonstrates a fundamental case of padding and tensor conversion for a list of NumPy array sequences. This assumes a fixed array shape within each sequence.

```python
import numpy as np
import torch

def pad_and_convert(list_of_lists, batch_first=False):
    """Pads a list of lists of NumPy arrays and converts it to a PyTorch tensor.

    Args:
        list_of_lists: A list of lists of NumPy arrays. Each inner list
                      represents a sequence.
        batch_first: If True, the output tensor will have batch as the first
                   dimension.

    Returns:
        A tuple containing the padded PyTorch tensor and the sequence lengths.
    """
    max_seq_len = max(len(seq) for seq in list_of_lists)
    sample_array = list_of_lists[0][0] #Assume consistent shape within the batch
    input_shape = sample_array.shape

    if batch_first:
        padded_tensor = torch.zeros(len(list_of_lists), max_seq_len, *input_shape)
    else:
        padded_tensor = torch.zeros(max_seq_len, len(list_of_lists), *input_shape)

    seq_lengths = []
    for i, seq in enumerate(list_of_lists):
       seq_len = len(seq)
       seq_lengths.append(seq_len)
       for j, array in enumerate(seq):
            if batch_first:
                padded_tensor[i, j] = torch.from_numpy(array)
            else:
                padded_tensor[j, i] = torch.from_numpy(array)

    return padded_tensor, torch.tensor(seq_lengths)

# Example Usage
data = [
    [np.random.rand(3, 5), np.random.rand(3, 5), np.random.rand(3, 5)],
    [np.random.rand(3, 5), np.random.rand(3, 5)],
    [np.random.rand(3, 5), np.random.rand(3, 5), np.random.rand(3, 5), np.random.rand(3, 5)]
]

padded_tensor, seq_lengths = pad_and_convert(data)
print("Padded tensor shape:", padded_tensor.shape)
print("Sequence lengths:", seq_lengths)

padded_tensor_batch_first, seq_lengths_bf = pad_and_convert(data, batch_first=True)
print("Padded tensor batch_first shape:", padded_tensor_batch_first.shape)
```

**Explanation of Code Example 1:**

The `pad_and_convert` function first computes the maximum sequence length within the provided list of lists. It then initializes a zero-filled PyTorch tensor of the appropriate shape based on the maximum sequence length, the batch size (number of lists), and the shape of the NumPy arrays within the sequences. The function then iterates through each sequence, copying the NumPy array into the padded tensor. It also creates a list, which contains the lengths of the sequences before padding. Finally, this function returns both the padded PyTorch tensor and a PyTorch tensor representing the original sequence lengths.  This length tensor is necessary for masking when you use a masked RNN. The example usage shows how to apply the function and inspect the output tensor’s shape and the length vector. We also demonstrate how batch_first parameter changes the output. The assumption here is that all NumPy arrays within each sequence have the same shape.

**Code Example 2: Handling Inconsistent Inner Shapes**

This example handles a case where the inner lists contain arrays with varying shapes. It assumes the sequences have a fixed number of arrays each, but each array can have a different shape. The padding is done on the *inner* array level, not on the *sequence* level. Each such array will be padded by 0s to ensure the same shape across the sequence.

```python
import numpy as np
import torch
from itertools import zip_longest

def pad_and_convert_inconsistent_inner(list_of_lists, batch_first=False):
    """Pads lists of NumPy arrays within a sequence and converts it to PyTorch tensor.
    Each array within a sequence can have different shape.

    Args:
        list_of_lists: A list of lists of NumPy arrays. Each inner list
                      represents a sequence.
        batch_first: If True, the output tensor will have batch as the first
                   dimension.
    Returns:
        A tuple containing the padded PyTorch tensor and the sequence lengths.
    """
    max_seq_len = max(len(seq) for seq in list_of_lists)
    #We get the shape of the biggest array, per index of each sequence.
    # The resulting variable max_inner_shapes is a list of tuples of shapes.
    max_inner_shapes = [
        max((x.shape for x in seq if x is not None), default=(0,))
        for seq in zip_longest(*list_of_lists)
    ]

    # Create the empty tensor.
    if batch_first:
        padded_tensor = torch.zeros(len(list_of_lists), max_seq_len, sum(x[0]*x[1] if x != (0,) else 0  for x in max_inner_shapes))
    else:
        padded_tensor = torch.zeros(max_seq_len, len(list_of_lists), sum(x[0]*x[1] if x != (0,) else 0  for x in max_inner_shapes))

    seq_lengths = []
    for i, seq in enumerate(list_of_lists):
        seq_len = len(seq)
        seq_lengths.append(seq_len)
        for j, array in enumerate(seq):
            if array is None:
                continue
            flat_array = torch.from_numpy(array.flatten())
            padding_size = max_inner_shapes[j][0] * max_inner_shapes[j][1]  - len(flat_array)
            flat_array = torch.nn.functional.pad(flat_array, (0,padding_size))
            if batch_first:
                padded_tensor[i, j] = flat_array
            else:
                padded_tensor[j, i] = flat_array

    return padded_tensor, torch.tensor(seq_lengths)


# Example Usage
data = [
    [np.random.rand(3, 5), np.random.rand(2, 7), np.random.rand(3, 2)],
    [np.random.rand(2, 3), np.random.rand(3, 3)],
    [np.random.rand(3, 3), np.random.rand(2, 2), np.random.rand(4, 3), np.random.rand(3, 1)]
]

padded_tensor, seq_lengths = pad_and_convert_inconsistent_inner(data)
print("Padded tensor shape:", padded_tensor.shape)
print("Sequence lengths:", seq_lengths)

padded_tensor_batch_first, seq_lengths_bf = pad_and_convert_inconsistent_inner(data, batch_first=True)
print("Padded tensor batch_first shape:", padded_tensor_batch_first.shape)
```

**Explanation of Code Example 2:**

In this case, the `pad_and_convert_inconsistent_inner` function first calculates the shape of the largest array in each position within the sequences using `zip_longest`. It ensures that each such array in the original sequences is then padded to the shape computed in `max_inner_shapes`.  It creates a padded tensor based on the sum of the flattened inner shapes and iterates through each sequence, flattening each array, padding it, and copying it to the correct position within the padded tensor. The core difference is handling variable shapes *inside* sequences instead of just varying sequence lengths.  The `zip_longest` from `itertools` is used to iterate over the lists, filling any missing values with `None`. This allows the code to process arrays when some sequences have fewer of the inner arrays than other sequences in the same batch.

**Code Example 3: Using Masking with PackedSequence**

This example combines padding with using a `PackedSequence`, which allows PyTorch to process sequences of variable lengths efficiently.  This is generally preferable to padding when dealing with LSTMs due to computational efficiency and improved handling of variable-length sequences. This technique does not require padding at the sequence level, but the code pads the inner arrays the same way that was done in code example 2.

```python
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from itertools import zip_longest

def pad_and_pack_inconsistent_inner(list_of_lists, batch_first=False):
    """Pads lists of NumPy arrays within a sequence, converts to pytorch tensor and then packs it using PackedSequence.

    Args:
        list_of_lists: A list of lists of NumPy arrays. Each inner list
                      represents a sequence.
        batch_first: If True, the output PackedSequence will have batch as the first
                   dimension.
    Returns:
        A packed sequence tensor.
    """
    max_inner_shapes = [
        max((x.shape for x in seq if x is not None), default=(0,))
        for seq in zip_longest(*list_of_lists)
    ]
    processed_seqs = []
    for seq in list_of_lists:
      padded_seq = []
      for j, array in enumerate(seq):
        if array is None:
           continue
        flat_array = torch.from_numpy(array.flatten())
        padding_size = max_inner_shapes[j][0] * max_inner_shapes[j][1] - len(flat_array)
        flat_array = torch.nn.functional.pad(flat_array, (0, padding_size))
        padded_seq.append(flat_array)
      if len(padded_seq) != 0:
          processed_seqs.append(torch.stack(padded_seq))

    packed_seq = pack_sequence(processed_seqs, enforce_sorted = False)
    return packed_seq



# Example Usage
data = [
    [np.random.rand(3, 5), np.random.rand(2, 7), np.random.rand(3, 2)],
    [np.random.rand(2, 3), np.random.rand(3, 3)],
    [np.random.rand(3, 3), np.random.rand(2, 2), np.random.rand(4, 3), np.random.rand(3, 1)]
]

packed_seq = pad_and_pack_inconsistent_inner(data)

print("Packed sequence:")
print(packed_seq)


padded_seq_out, lengths = pad_packed_sequence(packed_seq, batch_first=False)
print("Padded sequence output shape:", padded_seq_out.shape)
print("Sequence lengths:", lengths)


padded_seq_out_bf, lengths_bf = pad_packed_sequence(packed_seq, batch_first=True)
print("Padded sequence output batch first shape:", padded_seq_out_bf.shape)
print("Sequence lengths batch first:", lengths_bf)
```

**Explanation of Code Example 3:**

The `pad_and_pack_inconsistent_inner` function first pre-processes the sequences the same way as in example 2 by flattening the array and padding its flat version to the length specified by max inner shapes. Then, instead of building a padded tensor directly, it stacks the resulting padded arrays in each sequence into a `torch.Tensor` and it stacks the resulting sequences into a list `processed_seqs`. Finally, it calls PyTorch's `pack_sequence` function. This function converts list of sequences to a single sequence that groups the steps of all the batch elements together. This removes any padding operations during training. We can then reverse this process by calling `pad_packed_sequence` which outputs the padded tensor alongside the original sequence lengths. It also shows how the parameter `batch_first` influences the result.

**Resource Recommendations**

For further study on handling variable-length sequences in PyTorch, I suggest reviewing the official PyTorch documentation on RNNs, particularly the sections on `torch.nn.LSTM`, `torch.nn.utils.rnn.pack_sequence`, and `torch.nn.utils.rnn.pad_packed_sequence`. Additionally, I would look into tutorials covering advanced sequence modeling techniques and data preprocessing for NLP tasks since these typically involve handling variable sequence lengths. Reading documentation on `itertools.zip_longest` will be also beneficial.
