---
title: "How can a NumPy adjacency matrix and features be converted to a PyTorch Geometric data object?"
date: "2025-01-30"
id: "how-can-a-numpy-adjacency-matrix-and-features"
---
The critical hurdle in converting a NumPy adjacency matrix and feature matrix to a PyTorch Geometric (PyG) `Data` object lies in understanding PyG's expectation of data structure:  sparse adjacency matrices are generally preferred for efficiency, particularly with larger graphs.  My experience working on large-scale graph neural network projects highlighted this repeatedly; attempting to directly use dense adjacency matrices often leads to memory exhaustion and significantly slower processing times.

Therefore, the conversion process should prioritize creating a sparse representation of the adjacency matrix if it's not already sparse.  This significantly improves performance, especially for graphs with a low edge density.  The subsequent steps involve creating PyG-compatible tensors for both the adjacency information and node features, followed by assembling these into a `Data` object.

**1. Clear Explanation:**

The conversion process involves the following steps:

a) **Sparse Adjacency Matrix Representation:** If the adjacency matrix is dense (a NumPy array), convert it to a sparse representation using SciPy's `csr_matrix` function. This creates a Compressed Sparse Row (CSR) matrix, highly optimized for storage and computation in sparse graph scenarios.

b) **Tensor Conversion:** Convert the sparse adjacency matrix and the feature matrix (also assumed to be a NumPy array) into PyTorch tensors using `torch.tensor()`.  Ensure the data type is appropriate (usually `torch.float32` for features and `torch.long` for the adjacency matrix indices if using COO format).  Note that if you're converting a CSR matrix from SciPy, you should handle its components (values, row indices, column indices, and shape) individually.

c) **PyG `Data` Object Creation:** Instantiate a PyG `Data` object.  Populate this object with the tensors representing the edge indices and edge attributes (derived from the sparse matrix), node features, and optionally, node labels if available.


**2. Code Examples with Commentary:**

**Example 1:  Conversion from Dense Adjacency Matrix**

```python
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

# Sample data: Dense adjacency matrix and node features
dense_adj = np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [1, 0, 0, 1],
                     [0, 1, 1, 0]])
node_features = np.array([[1.0, 2.0],
                         [3.0, 4.0],
                         [5.0, 6.0],
                         [7.0, 8.0]])

# Convert to sparse matrix
sparse_adj = csr_matrix(dense_adj)

# Extract components of the sparse matrix
row, col = sparse_adj.nonzero()
edge_index = torch.tensor([row, col], dtype=torch.long)
edge_attr = torch.tensor(sparse_adj.data, dtype=torch.float) #Optional: If edges have attributes

# Convert features to PyTorch tensor
x = torch.tensor(node_features, dtype=torch.float)

# Create PyG Data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr) #edge_attr is optional

print(data)
```

This example explicitly demonstrates handling a dense adjacency matrix, converting it to a sparse representation, extracting relevant components, and subsequently creating a `Data` object.  The optional `edge_attr` showcases how to include edge features if present in the original data.


**Example 2: Conversion from Sparse Adjacency Matrix (COO format)**

```python
import numpy as np
import torch
from torch_geometric.data import Data

# Sample data: Sparse adjacency matrix (COO format) and node features
row = np.array([0, 1, 1, 2, 2])
col = np.array([1, 0, 3, 0, 3])
edge_index = torch.tensor([row, col], dtype=torch.long)
node_features = np.array([[1.0, 2.0],
                         [3.0, 4.0],
                         [5.0, 6.0],
                         [7.0, 8.0]])

# Convert features to PyTorch tensor
x = torch.tensor(node_features, dtype=torch.float)

# Create PyG Data object.  Edge attributes are omitted for brevity.
data = Data(x=x, edge_index=edge_index)

print(data)
```

This example begins with a sparse adjacency matrix already in Coordinate (COO) format, simplifying the conversion process. The focus is on directly using the edge indices and creating the `Data` object.  This is the most efficient approach when the initial data is already sparse.


**Example 3: Handling Node Labels**

```python
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

# Sample data with node labels
dense_adj = np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [1, 0, 0, 1],
                     [0, 1, 1, 0]])
node_features = np.array([[1.0, 2.0],
                         [3.0, 4.0],
                         [5.0, 6.0],
                         [7.0, 8.0]])
node_labels = np.array([0, 1, 0, 1])

# Convert to sparse matrix (same as Example 1)
sparse_adj = csr_matrix(dense_adj)
row, col = sparse_adj.nonzero()
edge_index = torch.tensor([row, col], dtype=torch.long)

# Convert features and labels to PyTorch tensors
x = torch.tensor(node_features, dtype=torch.float)
y = torch.tensor(node_labels, dtype=torch.long)

# Create PyG Data object including node labels
data = Data(x=x, edge_index=edge_index, y=y)

print(data)
```

This illustrates incorporating node labels (`y`) into the `Data` object. This is crucial for many graph tasks, such as node classification or graph classification.


**3. Resource Recommendations:**

The official PyTorch Geometric documentation.  The SciPy documentation on sparse matrices.  A good introductory textbook on graph theory and algorithms will provide valuable context.  Understanding linear algebra fundamentals, especially matrix representations, is also essential.  Finally, exploring example code within the PyTorch Geometric repository can be incredibly insightful.
