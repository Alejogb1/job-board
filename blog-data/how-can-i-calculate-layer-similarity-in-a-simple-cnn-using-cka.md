---
title: "How can I calculate layer similarity in a simple CNN using CKA?"
date: "2024-12-23"
id: "how-can-i-calculate-layer-similarity-in-a-simple-cnn-using-cka"
---

Alright, let's tackle layer similarity in a convolutional neural network using centered kernel alignment (CKA). This is a question I've encountered more times than I can count, often in the context of understanding feature representations across different stages of a network. It's not just about having a theoretical understanding; it's about seeing how different layers learn and what insights those similarities (or dissimilarities) might provide. I've personally used this in projects ranging from optimizing network architectures to debugging training issues that were otherwise quite elusive.

The core idea behind CKA is to compare two sets of data representations, effectively measuring the similarity of their underlying structures, without being sensitive to rotations, scaling, or other linear transformations. In this case, our "data representations" are the activations of different layers in the convolutional neural network when presented with an input batch.

Here's a breakdown of how to calculate CKA, along with python examples illustrating the process:

First, remember we're aiming to quantify the similarity between activations from two different layers. We are not comparing weights directly. CKA works by first computing a kernel matrix for each layer's activations. The kernel matrix measures the similarity between all pairs of data points *within* the activation space of that layer. Then, we compare these kernel matrices themselves.

Here’s a high-level overview of the steps:

1.  **Forward Pass:** Pass a batch of data through the network and obtain the activations of the layers you want to compare. We'll denote these as ‘x’ and ‘y’ for layers x and y respectively. This means taking intermediate outputs of the network before any pooling, activation functions or other changes to the tensor after specific convolutional layer.

2.  **Kernel Matrix Computation:** Compute a kernel matrix for each layer’s activations. A common and effective kernel to use here is the linear kernel (basically a dot product) which makes the code simpler and faster to execute for large activation tensors. If *xi* and *xj* are two activation vectors (flattened as necessary) from the same layer then the entry *Kij* of kernel matrix K is *xi^T xj*. Similarly, if *yi* and *yj* are two activation vectors (flattened as necessary) from the second layer, the matrix *L* has entries *Lij = yi^T yj*.

3.  **Centering the Kernel Matrix:** Before performing the CKA computation, we need to center the kernel matrices (K and L). Centering involves subtracting the mean of each row and each column from each element in the matrix, and adding the overall mean back to every element. This removes any bias in similarity measurements. A formula for doing this for matrix K is *Kcentered = K - 1n * K - K * 1n + 1n * K * 1n*. *1n* is a matrix of all ones of size nxn, where n is the number of data points.

4.  **CKA Calculation:** The CKA score between two centered kernel matrices, K and L, is then computed as: `CKA(K, L) = trace(K L^T) / sqrt(trace(K K^T) * trace(L L^T))`. The trace is the sum of diagonal elements. This CKA value will always be between 0 and 1, with 1 indicating identical similarity structure and 0 indicating maximum dissimilarity.

Here's a first code snippet demonstrating these steps, assuming the activations `x` and `y` are already obtained:

```python
import torch

def linear_kernel(x):
    return x @ x.T

def center_kernel(k):
    n = k.shape[0]
    unit = torch.ones(n, n).to(k.device)
    i = torch.eye(n).to(k.device)
    return k - unit @ k / n - k @ unit / n + unit @ k @ unit / (n * n)

def cka(k1, k2):
  k1_centered = center_kernel(k1)
  k2_centered = center_kernel(k2)
  cka_val = (k1_centered * k2_centered).sum()
  norm_k1 = torch.linalg.norm(k1_centered)
  norm_k2 = torch.linalg.norm(k2_centered)
  return (cka_val / (norm_k1 * norm_k2)).item()


# Assume x and y are obtained after a forward pass.
# Each has shape [batch_size, *feature_dims]

def compute_cka_between_layers(x,y):
  x_flat = x.flatten(start_dim=1)
  y_flat = y.flatten(start_dim=1)

  k1 = linear_kernel(x_flat)
  k2 = linear_kernel(y_flat)

  return cka(k1,k2)

#example Usage
if __name__ == '__main__':

    batch_size=10
    feature_dim1 = 64*8*8
    feature_dim2 = 128*4*4

    x = torch.randn(batch_size,feature_dim1)
    y = torch.randn(batch_size,feature_dim2)

    cka_value = compute_cka_between_layers(x,y)
    print(f"The computed CKA value between the layers: {cka_value}")
```

This first example demonstrates the core logic. However, sometimes we need to perform this across more than just two layers. Let's say you want to compare *all* pairwise layer activations within the network. I had a situation once where visualizing the heatmap of pairwise CKA values revealed a crucial bottleneck in information flow, allowing me to quickly diagnose and rectify the issue.

Here’s the second code example that demonstrates how to calculate pairwise CKA between multiple layers:

```python
import torch
import torch.nn as nn

def linear_kernel(x):
    return x @ x.T

def center_kernel(k):
    n = k.shape[0]
    unit = torch.ones(n, n).to(k.device)
    i = torch.eye(n).to(k.device)
    return k - unit @ k / n - k @ unit / n + unit @ k @ unit / (n * n)

def cka(k1, k2):
    k1_centered = center_kernel(k1)
    k2_centered = center_kernel(k2)
    cka_val = (k1_centered * k2_centered).sum()
    norm_k1 = torch.linalg.norm(k1_centered)
    norm_k2 = torch.linalg.norm(k2_centered)
    return (cka_val / (norm_k1 * norm_k2)).item()


def compute_cka_matrix_for_layers(activations):
  num_layers = len(activations)
  cka_matrix = torch.zeros(num_layers, num_layers)

  for i in range(num_layers):
    for j in range(num_layers):

        x_flat = activations[i].flatten(start_dim=1)
        y_flat = activations[j].flatten(start_dim=1)

        k1 = linear_kernel(x_flat)
        k2 = linear_kernel(y_flat)

        cka_matrix[i,j]=cka(k1,k2)

  return cka_matrix



#example Usage
if __name__ == '__main__':

  class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(128,256,kernel_size=3, padding = 1)
        self.fc = nn.Linear(256 * 8 * 8, 10)

    def forward(self, x):
        activations = []
        x = self.relu(self.conv1(x))
        activations.append(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        activations.append(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        activations.append(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,activations


    batch_size = 10
    input_size = (batch_size,3,32,32)

    cnn_model = SimpleCNN()

    input = torch.randn(input_size)
    _, layer_activations = cnn_model(input)
    cka_matrix = compute_cka_matrix_for_layers(layer_activations)
    print("Computed CKA matrix: \n",cka_matrix)
```

Finally, an important consideration, especially when using larger models, is computational efficiency. Calculating kernel matrices for very large activations can be memory intensive. In one particular case, processing activations from an extremely deep model caused an out of memory error until I realized that CKA values don't change drastically when a relatively small but representative subset of the total dataset is used. So the next code snippet illustrates that:

```python
import torch
import torch.nn as nn

def linear_kernel(x):
    return x @ x.T

def center_kernel(k):
    n = k.shape[0]
    unit = torch.ones(n, n).to(k.device)
    i = torch.eye(n).to(k.device)
    return k - unit @ k / n - k @ unit / n + unit @ k @ unit / (n * n)

def cka(k1, k2):
    k1_centered = center_kernel(k1)
    k2_centered = center_kernel(k2)
    cka_val = (k1_centered * k2_centered).sum()
    norm_k1 = torch.linalg.norm(k1_centered)
    norm_k2 = torch.linalg.norm(k2_centered)
    return (cka_val / (norm_k1 * norm_k2)).item()


def compute_cka_matrix_for_layers_subset(activations,subset_size):
  num_layers = len(activations)
  cka_matrix = torch.zeros(num_layers, num_layers)
  batch_size = activations[0].shape[0]

  if subset_size > batch_size:
    subset_size = batch_size

  subset_indices = torch.randperm(batch_size)[:subset_size]


  for i in range(num_layers):
    for j in range(num_layers):

        x_subset = activations[i][subset_indices].flatten(start_dim=1)
        y_subset = activations[j][subset_indices].flatten(start_dim=1)

        k1 = linear_kernel(x_subset)
        k2 = linear_kernel(y_subset)
        cka_matrix[i,j]=cka(k1,k2)

  return cka_matrix



#example Usage
if __name__ == '__main__':

  class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(128,256,kernel_size=3, padding = 1)
        self.fc = nn.Linear(256 * 8 * 8, 10)

    def forward(self, x):
        activations = []
        x = self.relu(self.conv1(x))
        activations.append(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        activations.append(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        activations.append(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,activations

  batch_size = 100
  subset_size = 20
  input_size = (batch_size,3,32,32)

  cnn_model = SimpleCNN()

  input = torch.randn(input_size)
  _, layer_activations = cnn_model(input)

  cka_matrix_subset = compute_cka_matrix_for_layers_subset(layer_activations,subset_size)
  print("Computed CKA matrix using subset: \n",cka_matrix_subset)

```
As for resources, I would recommend checking out the original CKA paper, "Similarity of Neural Network Representations Revisited" by Kornblith et al. (available on arXiv). This will provide you with a deeper dive into the theoretical underpinnings.  For practical implementations, the "Interpretable Machine Learning" book by Christoph Molnar, though not directly about CKA, covers representation similarity concepts well and links them to interpretability practices.

Hopefully this helps clarify the process, and gives you a good starting point for using CKA in your own projects. It’s a powerful tool for understanding the inner workings of your neural networks.
