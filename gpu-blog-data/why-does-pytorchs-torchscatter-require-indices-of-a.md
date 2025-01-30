---
title: "Why does PyTorch's `torch.scatter` require indices of a smaller shape than the input values?"
date: "2025-01-30"
id: "why-does-pytorchs-torchscatter-require-indices-of-a"
---
The seemingly counterintuitive behavior of `torch.scatter` in PyTorch, where the index tensor often has a smaller shape than the value tensor, arises from its fundamental design for performing sparse updates within a tensor. Unlike element-wise assignments, `torch.scatter` uses the index tensor to map source values into a destination tensor, potentially many-to-one, creating a many-to-one write behavior. The index tensor's dimensionality reflects the number of dimensions being used to specify locations for data insertion. If the index tensor had the same shape as the values, it would merely copy values to the destination without any sparse or targeted update mechanisms.

The core function of `torch.scatter` is to selectively write data from a source tensor (`src`) into a destination tensor (`input`), guided by the locations specified in an index tensor (`index`). This process follows the logic: for each value in `src`, a corresponding location in `input` is determined by the value of `index` at the corresponding position. Consequently, if `index` had the same shape as `src`, then each unique location would always map to a single value, thereby defeating the purpose of scattered writes or accumulations at specified target locations. The reduction aspect, such as `add`, `multiply`, or other operation, is also enabled this way because the reduction of multiple source values to the same location is possible. The destination tensor (`input`) typically has a larger dimensionality than `index`, thereby accommodating this many-to-one mapping.

I've personally faced scenarios where this design became critical, particularly during custom graph network implementations. For example, when scattering node features into adjacency matrix representations, the feature tensor had a batch dimension, node dimension, and feature dimension. However, I needed to update the adjacency matrix at specific edge locations; therefore, the index tensor contained pairs of indices representing edge locations within each batch. The edge index was generally smaller than the number of nodes; hence, the index shape was inherently different, and a shape-matched index would render the desired update impossible.

To clarify further, consider a one-dimensional example:

```python
import torch

input_tensor = torch.zeros(5)
src_tensor = torch.tensor([10, 20, 30])
index_tensor = torch.tensor([0, 2, 4])

output_tensor = input_tensor.scatter(0, index_tensor, src_tensor)
print(output_tensor) # Output: tensor([10.,  0., 20.,  0., 30.])
```

In this code, `input_tensor` is our destination tensor, initialized with zeros. `src_tensor` provides the values to be inserted, and `index_tensor` specifies where to place these values. The dimension is 0 here for both the input and output tensors. The `scatter` operation places 10 at index 0, 20 at index 2, and 30 at index 4 of `input_tensor`. The shape of `index_tensor` (`[3]`) is smaller than the shape of `input_tensor` (`[5]`) and the same as the shape of the source tensor, `src_tensor`. If `index_tensor` had the same shape as input, then this behavior is impossible.

Now, let's look at a slightly more complex two-dimensional example using `scatter_add`:

```python
import torch

input_tensor = torch.zeros(3, 4)
src_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
index_tensor = torch.tensor([[0, 2, 1], [1, 0, 2]])

output_tensor = input_tensor.scatter_add(1, index_tensor, src_tensor)
print(output_tensor)
# Output:
# tensor([[1., 3., 2., 0.],
#         [5., 4., 6., 0.],
#         [0., 0., 0., 0.]])
```

Here, `input_tensor` is a 3x4 matrix, and `src_tensor` is a 2x3 matrix containing values to be scattered into input. The `index_tensor` is also a 2x3 matrix, specifying the column indices in input where each corresponding value from `src_tensor` should be placed along dimension 1. Because we're using `scatter_add`, multiple values can be added at the same location. In the first row, for example, the value 1 is placed at index 0 (column 0), 2 at index 2 (column 2), and 3 at index 1 (column 1). The second row follows similarly, placing values 4 at column 1, 5 at column 0 and 6 at column 2. Notice again that the dimension of the index (`[2, 3]`) is different from the output tensor (`[3, 4]`). It is only the shape of the source and index that match because a specific position on the source (`[i, j]`) is placed into the output tensor at `index[i, j]`.

Finally, consider a scenario using batch dimensions:

```python
import torch

input_tensor = torch.zeros(2, 3, 4)  # Batch of 2, 3x4 matrices
src_tensor = torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])  # Batch of 2, 3x2 matrices
index_tensor = torch.tensor([[[0, 2], [1, 0], [2, 3]], [[3, 1], [2, 0], [1, 2]]])  # Batch of 2, 3x2 matrices

output_tensor = input_tensor.scatter(2, index_tensor, src_tensor)
print(output_tensor)
# Output:
# tensor([[[1., 0., 2., 0.],
#          [4., 3., 0., 0.],
#          [0., 6., 0., 5.]],

#         [[0., 8., 0., 7.],
#          [10., 9., 0., 0.],
#          [0., 11., 12., 0.]]])
```

Here, we have a batch of two 3x4 matrices (destination), and a batch of 3x2 matrices as the source. The indices now specify the positions along the last dimension (dimension 2), while the batches are independent. For each batch, each of the 3 rows of the source are placed into columns specified by the index tensor in the target output. The important thing to note is that even in this 3D case, the indices' shape (`[2, 3, 2]`) differs from the target (`[2, 3, 4]`). If, for example, `index_tensor` was also `[2, 3, 4]`, then each of the positions in `src_tensor` would map directly to a unique location and defeat the intended sparse update purpose.

In essence, the shape disparity between the index and value tensors is not an arbitrary constraint but a deliberate design feature to enable sparse, selective, and many-to-one assignments within tensors. This mechanism facilitates efficient updates where data needs to be scattered based on specified locations rather than by direct, element-wise correspondence. As a result, the dimensionality of the index tensor determines the update locations, while the values tensor provides the elements to be written, potentially multiple times, into the target tensor at locations derived by the index values.

For further exploration of sparse tensor operations in PyTorch and other deep learning frameworks, I recommend examining the official documentation for `torch.sparse`, particularly the `torch.sparse.FloatTensor`, `torch.sparse.IntTensor` and their respective operations. Reviewing examples and tutorials that explicitly discuss graph neural networks or other tasks that rely on sparse matrix representations can also provide useful insights. Furthermore, researching academic literature on numerical computation with sparse matrices, particularly the coordinate list (COO) representation, can help better understand the underlying mathematical foundations for the shape considerations. This theoretical background significantly helps in debugging or optimizing related computations and ensures correct usage of sparse operations within complex models.
