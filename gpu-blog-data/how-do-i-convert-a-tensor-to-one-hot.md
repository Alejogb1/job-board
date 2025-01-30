---
title: "How do I convert a tensor to one-hot encoding?"
date: "2025-01-30"
id: "how-do-i-convert-a-tensor-to-one-hot"
---
The inherent structure of one-hot encoding, where each categorical value is represented by a vector with a single '1' at the index corresponding to the category and '0's elsewhere, presents a common challenge in machine learning pipelines. My experience in developing anomaly detection systems for sensor data, particularly in scenarios involving categorical sensor types, has highlighted the need for efficient and correct one-hot encoding methods. The transformation from a tensor containing categorical indices to its one-hot representation is critical for ensuring the input compatibility of various machine learning models.

The process involves generating a multi-dimensional array (a new tensor) where each row corresponds to an element in the original tensor, and each column represents a unique category. This array contains binary values indicating the presence or absence of the associated category. Crucially, this conversion requires knowing the total number of possible categories beforehand. This knowledge determines the dimensionality of the one-hot encoded vectors. Failure to correctly establish this dimensionality will lead to misrepresentation of the categorical data.

For a tensor of shape `(N,)`, where N is the number of categorical observations, the goal is to convert it into a tensor of shape `(N, C)`, where C is the number of categories. If the original tensor has shape `(B, N)`, where B is the batch size, the one-hot tensor will correspondingly have shape `(B, N, C)`. The underlying principle across all dimensions is the same: replace each integer representation with a binary vector. This transformation enables the application of numerous machine learning techniques, especially those designed for numeric data representation.

Here are three examples using Python with the PyTorch library, which I've found versatile for this task.

**Example 1: One-Hot Encoding a 1D Tensor**

```python
import torch

# Example tensor of categorical indices.
labels = torch.tensor([2, 0, 1, 2, 0])
num_classes = 3  # Number of unique categories

# Use torch.nn.functional.one_hot to create one-hot encoding.
one_hot_encoded = torch.nn.functional.one_hot(labels, num_classes=num_classes)

print("Original labels:")
print(labels)
print("\nOne-hot encoded tensor:")
print(one_hot_encoded)

```

*Commentary:* In this basic example, `torch.nn.functional.one_hot` is the workhorse. It accepts the input tensor and the number of distinct categories as parameters. The method returns the one-hot representation, automatically creating a new tensor with the correct dimensions, where each row is the one-hot vector for each category. The `labels` variable represents the category indices. The `num_classes` specifies that there are three different possible classes (indexed as 0, 1, and 2). The output demonstrates that each integer in the original tensor is mapped to a one-hot vector of length three. This operation is efficient for simple lists or 1D tensors of categorical data.

**Example 2: One-Hot Encoding a 2D Tensor (Batch Processing)**

```python
import torch

# Example batch of categorical indices, 2 batches of 3 elements
batch_labels = torch.tensor([[1, 0, 2], [2, 1, 0]])
num_classes = 3 # Number of unique categories

# Convert the labels to long type as one_hot requires this
batch_labels = batch_labels.long()
# Apply one_hot to the batch of indices.
batch_one_hot = torch.nn.functional.one_hot(batch_labels, num_classes=num_classes)

print("Original batch labels:")
print(batch_labels)
print("\nOne-hot encoded batch tensor:")
print(batch_one_hot)

```

*Commentary:*  This example illustrates the usage with a batch of samples. A tensor with two dimensions, where the first dimension represents the batch size, is provided. The function still operates element-wise along the second dimension, creating one-hot vectors for every index in the batch. The key difference here is that `torch.nn.functional.one_hot` automatically handles the batch dimension, adding another dimension to represent the one-hot vectors. The batch is of size (2,3), and the resulting one-hot tensor is (2,3,3), meaning two batches, each containing three elements that are now one-hot encoded with 3 classes. I've added `.long()` method to ensure the input tensor is in `torch.long` datatype because `torch.nn.functional.one_hot` expects indices to be of integer type. This is a common step in data preparation when dealing with tensors of type float.

**Example 3: Custom Function for Handling Unknown Categories**

```python
import torch

def one_hot_encode_custom(labels, num_classes):
    """Custom one-hot encoder handling cases outside known classes"""

    one_hot = torch.zeros(labels.size(0), num_classes) # allocate zeros
    
    # Filter labels within range, outside will get all zeros
    mask = (labels >= 0) & (labels < num_classes)

    indices = torch.arange(labels.size(0))[mask]
    label_values = labels[mask]
    one_hot[indices, label_values] = 1

    return one_hot

# Example tensor with some out of range class indices
labels_custom = torch.tensor([2, 0, 1, 3, -1, 0])
num_classes_custom = 3

one_hot_encoded_custom = one_hot_encode_custom(labels_custom, num_classes_custom)

print("Original labels with outliers:")
print(labels_custom)
print("\nCustom one-hot encoded tensor:")
print(one_hot_encoded_custom)

```

*Commentary:*  This example demonstrates a custom implementation of one-hot encoding, primarily to handle situations where indices might fall outside the anticipated range (for instance, a label value of '3' when you only have categories '0', '1', and '2'). This is relevant because, in real-world scenarios, data can be noisy or may contain unanticipated values. The custom function first initializes a tensor of zeros and then iteratively sets the correct indices to 1 based on the provided label values while masking any labels that are out of range of defined classes.  Note that in this version I have assumed negative indices are invalid labels, this can easily be adapted. This approach provides more control, such as ignoring or flagging out-of-bounds values during preprocessing. This technique avoids common runtime errors associated with using direct `one_hot` functions on raw, un-validated data.

For further exploration of this topic and broader considerations in data preprocessing for machine learning, I suggest consulting texts that focus on data representation and deep learning techniques. Sources discussing data preparation for neural networks, as well as textbooks detailing fundamentals of linear algebra would be useful. Additionally, practical guides detailing the application of libraries such as PyTorch and TensorFlow can provide detailed examples and use cases. I've found resources on categorical embeddings to be also beneficial. Exploring the internal workings of libraries implementing one-hot encoding will contribute to a deeper understanding of underlying principles.
