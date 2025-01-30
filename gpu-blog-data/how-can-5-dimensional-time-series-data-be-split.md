---
title: "How can 5-dimensional time series data be split and embedded, selectively passing some dimensions through an embedding layer?"
date: "2025-01-30"
id: "how-can-5-dimensional-time-series-data-be-split"
---
Handling 5-dimensional time series data presents unique challenges when integrating with neural network architectures, particularly when selectively applying embedding layers. In my experience building anomaly detection systems for complex manufacturing processes, we often encountered data like this, consisting of time series readings from five independent sensors (think temperature, pressure, vibration along three axes) at consistent intervals. The key here isn't simply treating these dimensions uniformly; some dimensions possess inherent ordinality or represent values that, when viewed directly, are more informative than when processed through a black-box embedding.

The core issue revolves around efficiently preparing the data for consumption by a neural network where some, but not all, dimensions require transformation via an embedding layer. Traditional approaches, like using a single embedding for the entire 5-dimensional input or manually flattening the input without any sophisticated manipulation, often fail to capitalize on the distinct characteristics of individual features. Instead, we need a strategy that splits the data, processes some dimensions through embedding layers, and concatenates the results before further processing. I've found that focusing on a modular approach yields the most robust and adaptable solution.

The basic concept involves treating our input as a sequence of five feature vectors, each containing scalar data at a specific time point. We'll selectively apply an embedding only to specific dimensions and, crucially, this process should be trainable end-to-end. This requires careful handling of tensors and the design of a model that can accept both embedded and non-embedded input dimensions. The strategy I generally deploy consists of three key steps. First, we logically separate the dimensions into two groups: those which will pass directly through the model (direct input) and those which will be transformed with an embedding layer. Second, we construct embeddings for the dimensions which require it. Third, we concatenate both the embedded and non-embedded results and feed them into subsequent network layers.

Here is a detailed code example implementing this approach using PyTorch:

```python
import torch
import torch.nn as nn

class TimeSeriesPreprocessor(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, direct_input_dims):
        super(TimeSeriesPreprocessor, self).__init__()
        self.direct_input_dims = direct_input_dims # Store indices of non-embedded dimensions.
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        # Input x is expected to have shape (batch_size, sequence_length, 5)

        batch_size, seq_len, _ = x.shape
        embedded_features = []
        direct_features = []

        # Process each dimension.
        for dim_idx in range(x.shape[2]):
             if dim_idx not in self.direct_input_dims:
                 dim_input = x[:, :, dim_idx].long()  # Assumes integer-encoded categories
                 dim_embed = self.embedding(dim_input) # Shape: (batch_size, seq_len, embedding_dim)
                 embedded_features.append(dim_embed)
             else:
                 dim_input = x[:, :, dim_idx].float().unsqueeze(-1) # Shape: (batch_size, seq_len, 1)
                 direct_features.append(dim_input)

        # Concatenate embedded and non-embedded features.
        if embedded_features and direct_features:
            all_features = torch.cat(embedded_features + direct_features, dim=2)
        elif embedded_features:
            all_features = torch.cat(embedded_features, dim=2)
        else:
            all_features = torch.cat(direct_features, dim=2)


        return all_features


# Example usage:

num_embeddings = 10 # Example for a categorical sensor
embedding_dim = 8 # Example embedding dimension
direct_input_dims = [0, 3]  # Dimensions 0 and 3 will bypass embedding
sequence_length = 20
batch_size = 32
input_shape = (batch_size, sequence_length, 5)

# Generate some mock data
input_data = torch.randint(0, num_embeddings, input_shape).float()

preprocessor = TimeSeriesPreprocessor(num_embeddings, embedding_dim, direct_input_dims)
processed_data = preprocessor(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {processed_data.shape}")
```

This first example demonstrates the fundamental logic. We create a `TimeSeriesPreprocessor` class that encapsulates the selective embedding process. The constructor takes the number of embeddings, embedding dimensions, and a list of indices corresponding to dimensions that should bypass the embedding and be directly used as part of the model's input. In the `forward` method, the code iterates through each dimension of the input data. Dimensions specified in `direct_input_dims` are treated as floating-point values, added to the `direct_features` list after being expanded to a three-dimensional tensor, and the others, as integers and will pass through the embedding layer. After processing, these lists of feature tensors are concatenated along the feature dimension (axis 2), constructing the final representation.  The output shape from the example clearly shows how the dimensionality has changed after this process. For example, if only the 0th and 3rd dimensions are passed through without embeddings (and are therefore single float vectors), and the embedding dimension is 8, then the output will have a shape of (32, 20, (8*3)+1+1), which evaluates to (32, 20, 26).

The next example below showcases a more generalized approach suitable for scenarios where some direct input dimensions should receive further processing:

```python
import torch
import torch.nn as nn

class AdvancedTimeSeriesPreprocessor(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, direct_input_dims, direct_processing_dims, direct_processing_size):
        super(AdvancedTimeSeriesPreprocessor, self).__init__()
        self.direct_input_dims = direct_input_dims
        self.direct_processing_dims = direct_processing_dims
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.linear_layers = nn.ModuleList([nn.Linear(1, direct_processing_size) if dim_idx in direct_processing_dims else None
                                        for dim_idx in range(5)])


    def forward(self, x):
         # Input x is expected to have shape (batch_size, sequence_length, 5)

         embedded_features = []
         direct_features = []

         for dim_idx in range(x.shape[2]):
            if dim_idx not in self.direct_input_dims:
                 dim_input = x[:, :, dim_idx].long()
                 dim_embed = self.embedding(dim_input)
                 embedded_features.append(dim_embed)
            else:
                 dim_input = x[:, :, dim_idx].float().unsqueeze(-1)

                 if self.linear_layers[dim_idx] is not None:
                     dim_processed = self.linear_layers[dim_idx](dim_input)
                     direct_features.append(dim_processed)
                 else:
                     direct_features.append(dim_input)

         if embedded_features and direct_features:
             all_features = torch.cat(embedded_features + direct_features, dim=2)
         elif embedded_features:
             all_features = torch.cat(embedded_features, dim=2)
         else:
             all_features = torch.cat(direct_features, dim=2)

         return all_features

# Example usage:
num_embeddings = 10
embedding_dim = 8
direct_input_dims = [0, 3]
direct_processing_dims = [0] # Only the 0th dimension from the direct inputs is processed
direct_processing_size = 4
sequence_length = 20
batch_size = 32
input_shape = (batch_size, sequence_length, 5)

# Generate some mock data
input_data = torch.randint(0, num_embeddings, input_shape).float()

preprocessor = AdvancedTimeSeriesPreprocessor(num_embeddings, embedding_dim, direct_input_dims, direct_processing_dims, direct_processing_size)
processed_data = preprocessor(input_data)


print(f"Input shape: {input_data.shape}")
print(f"Output shape: {processed_data.shape}")
```

In this refined version, `AdvancedTimeSeriesPreprocessor`, we add `direct_processing_dims` and `direct_processing_size` parameters. This allows specifying certain direct input dimensions that are passed through a linear layer to be transformed to a specified dimension. In the example, only the 0th dimension is passed through a linear transformation. The `forward` method now checks if a dimension is in the `direct_processing_dims`, applies the appropriate transformation if needed using the `nn.Linear` layer, and concatenates all processed dimensions before the final return. This allows us to capture potentially meaningful transformations of non-embedded dimensions and thus improve model performance. The output shape would now be (32, 20, (8 * 3) + 4 + 1) == (32, 20, 29) in this example.

Finally, the following code snippet demonstrates a completely general approach utilizing dynamic embedding assignment, further improving versatility:

```python
import torch
import torch.nn as nn

class DynamicPreprocessor(nn.Module):
    def __init__(self, embedding_configs, direct_processing_sizes):
         super(DynamicPreprocessor, self).__init__()
         self.embedding_modules = nn.ModuleList()
         self.linear_modules = nn.ModuleList()


         for config in embedding_configs:
             if config is None:
                 self.embedding_modules.append(None)
             else:
                 self.embedding_modules.append(nn.Embedding(config[0],config[1]))


         for i in range(len(direct_processing_sizes)):
              if direct_processing_sizes[i] is None:
                 self.linear_modules.append(None)
              else:
                 self.linear_modules.append(nn.Linear(1, direct_processing_sizes[i]))


    def forward(self, x):
         # Input x is expected to have shape (batch_size, sequence_length, 5)
          processed_features = []

          for dim_idx in range(x.shape[2]):
             dim_input = x[:, :, dim_idx]

             if self.embedding_modules[dim_idx] is not None:
                 dim_input = dim_input.long()
                 dim_embed = self.embedding_modules[dim_idx](dim_input)
                 processed_features.append(dim_embed)
             elif self.linear_modules[dim_idx] is not None:
                 dim_input = dim_input.float().unsqueeze(-1)
                 dim_transformed = self.linear_modules[dim_idx](dim_input)
                 processed_features.append(dim_transformed)
             else:
                 dim_input = dim_input.float().unsqueeze(-1)
                 processed_features.append(dim_input)
          all_features = torch.cat(processed_features, dim=2)

          return all_features


# Example Usage:
embedding_configs = [
    (10,8),  # Embedding for dimension 0 (10 embeddings, dimension 8)
    None, # No embedding for dimension 1
    (12,4), # Embedding for dimension 2 (12 embeddings, dimension 4)
    None,  # No embedding for dimension 3
    None # No embedding for dimension 4

]
direct_processing_sizes = [
   None,
   2,
   None,
   3,
   None
] #Linear transform dimensions.

sequence_length = 20
batch_size = 32
input_shape = (batch_size, sequence_length, 5)

input_data = torch.randint(0, 15, input_shape).float() # Example Input
preprocessor = DynamicPreprocessor(embedding_configs, direct_processing_sizes)
processed_data = preprocessor(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {processed_data.shape}")
```

`DynamicPreprocessor` offers the greatest flexibility by accepting a list of tuples, where each tuple represents the number of embeddings and the embedding dimension for each feature, or `None` if no embedding is needed, as well as a direct processing list where the value is either `None` if not direct processing is needed, or the dimension to which the feature should be mapped by linear transformation. This allows complete configuration of dimension processing without needing code modifications. The `forward` method iterates through each dimension, dynamically selecting either the appropriate embedding layer, linear transform, or passing the input directly based on the provided configurations. The output of this final example would be of the form (32, 20, 8+1+4+3+1) == (32, 20, 17)

For further study, I would recommend exploring resources that cover sequence modeling, particularly the concepts of recurrent neural networks and transformers, as these often use embedding layers in similar ways. Examining texts focusing on the PyTorch deep learning framework itself can provide an in-depth understanding of data handling techniques within the framework. Finally, articles concerning multivariate time series analysis and feature engineering could be highly beneficial for those looking to further refine their methods.
