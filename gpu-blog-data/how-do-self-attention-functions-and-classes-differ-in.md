---
title: "How do self-attention functions and classes differ in implementation?"
date: "2025-01-30"
id: "how-do-self-attention-functions-and-classes-differ-in"
---
In my experience developing custom transformer models for time-series forecasting, I’ve observed that while the theoretical basis of self-attention remains consistent, its practical implementation differs significantly between functional approaches and object-oriented classes. The choice profoundly affects modularity, reusability, and computational overhead within a larger machine learning pipeline.

Fundamentally, self-attention calculates a weighted sum of input representations, allowing each element in a sequence to attend to other elements. This process involves three primary learned linear transformations of the input sequence, typically denoted as queries (Q), keys (K), and values (V). The attention weights are derived by applying a scaled dot product between Q and K, followed by a softmax activation. These weights are then applied to V to produce the final attended output. The core calculation is the same whether implemented as a function or a class, yet the manner in which state is handled and the scope of applicability varies substantially.

A functional approach encapsulates the self-attention logic within a standalone function. This often involves accepting the input sequence and its associated query, key, and value matrices as arguments, along with other necessary parameters like the masking tensor or the number of attention heads. All computations are executed within the scope of the function, and the result is directly returned. This method emphasizes simplicity and conciseness, allowing for the rapid prototyping of models. The function itself is stateless, with no persistent information beyond the execution context of each call. It promotes immutability and can be easily integrated into existing functional programming paradigms.

Classes, conversely, implement self-attention as an object encapsulating both the logic and parameters of the operation. A class-based self-attention module initializes the necessary linear transformation layers (query, key, value projections) as its internal state. These linear transformations often use weight initialization schemes and can maintain batch-normalized weights, along with masking strategies that may need to be defined during construction. The forward pass, executed via a dedicated `forward()` method, processes input sequences by applying these transformations and performing the core attention mechanism. This approach facilitates more complex behavior, allowing methods for parameter management, customized masking strategies, and potentially integration with optimizers and other training functionalities. Importantly, a class-based implementation enables instantiation and reuse in multiple parts of a model and can be treated as an individual component.

To exemplify, let’s consider a simplified functional implementation using a hypothetical tensor library (akin to PyTorch or TensorFlow). This code demonstrates the essential attention calculation within a single function, `self_attention_function`.

```python
import hypothetical_tensor_lib as hl

def self_attention_function(query, key, value, mask=None):
    """
    Computes self-attention using a functional approach.

    Args:
      query: Tensor of shape (batch_size, sequence_length, dimension_q).
      key: Tensor of shape (batch_size, sequence_length, dimension_k).
      value: Tensor of shape (batch_size, sequence_length, dimension_v).
      mask: Optional boolean mask for attended elements.

    Returns:
      Tensor of shape (batch_size, sequence_length, dimension_v).
    """

    dimension_k = key.shape[-1] # Infer the dimension from the key tensor

    attention_weights = hl.matmul(query, hl.transpose(key, -1, -2)) # Dot product Q and K^T
    attention_weights = attention_weights / hl.sqrt(dimension_k)  # Scaled dot product

    if mask is not None:
        attention_weights = hl.masked_fill(attention_weights, mask == 0, -1e9)  # Masking

    attention_weights = hl.softmax(attention_weights, axis=-1)   # Softmax activation
    output = hl.matmul(attention_weights, value)                  # Weighted sum of V
    return output
```

This function `self_attention_function` directly computes the self-attention output from provided Q, K, V matrices. The function is stateless and each call operates independently. It would require the caller to handle the linear transformations that generate Q, K, and V. The advantage is its simplicity and clear representation of the core mechanism. If different model sections require differing parameter initialization or masking strategies it could be an issue.

Contrast this with a class-based approach. The following `SelfAttentionLayer` class encapsulates the linear transformations and the entire attention calculation:

```python
import hypothetical_tensor_lib as hl

class SelfAttentionLayer:
  def __init__(self, dimension, num_heads, masking_strategy="none"):
    """
        Initializes the self-attention layer.

        Args:
          dimension: Input dimension.
          num_heads: The number of attention heads.
          masking_strategy: The type of masking to use "none", "causal", or a "custom" mask.
    """

    self.dimension = dimension
    self.num_heads = num_heads
    self.head_dimension = dimension // num_heads # Individual head dimension
    assert dimension % num_heads == 0, "Dimension must be divisible by num_heads."

    # Linear transformation layers for Q, K, V, initialized based on a default he scheme
    self.query_projection = hl.LinearLayer(dimension, dimension)
    self.key_projection   = hl.LinearLayer(dimension, dimension)
    self.value_projection = hl.LinearLayer(dimension, dimension)

    self.output_projection = hl.LinearLayer(dimension, dimension) # Layer used for output transformation
    self.masking_strategy = masking_strategy # Parameter to be set when constructing self-attention layer

  def forward(self, x, mask=None):
    """
        Performs forward pass of self-attention layer

        Args:
          x: Input tensor of shape (batch_size, sequence_length, dimension).
          mask: Optional masking tensor for attended elements.

        Returns:
          Tensor of shape (batch_size, sequence_length, dimension).
    """
    batch_size, sequence_length, _ = x.shape # retrieve dimensions

    query = self.query_projection(x)
    key = self.key_projection(x)
    value = self.value_projection(x)

    # Reshape for multi-head
    query = hl.reshape(query, (batch_size, sequence_length, self.num_heads, self.head_dimension))
    key   = hl.reshape(key,   (batch_size, sequence_length, self.num_heads, self.head_dimension))
    value = hl.reshape(value, (batch_size, sequence_length, self.num_heads, self.head_dimension))

    query = hl.transpose(query, 1, 2) # Transpose sequence length and number of heads
    key   = hl.transpose(key,   1, 2)
    value = hl.transpose(value, 1, 2)

    attention_weights = hl.matmul(query, hl.transpose(key, -1, -2)) #Dot product of Q and K^T
    attention_weights = attention_weights / hl.sqrt(self.head_dimension)

    if self.masking_strategy == "causal": # Applies causal masking
      causal_mask = hl.tril(hl.ones((sequence_length, sequence_length)), diagonal=0).to(bool)
      attention_weights = hl.masked_fill(attention_weights, causal_mask == 0, -1e9)

    elif mask is not None:
        attention_weights = hl.masked_fill(attention_weights, mask == 0, -1e9) # Applies a custom mask

    attention_weights = hl.softmax(attention_weights, axis=-1) # Softmax operation
    output = hl.matmul(attention_weights, value)    # Weighted sum of V

    # Undo the multi-head reshaping and transpose
    output = hl.transpose(output, 1, 2)
    output = hl.reshape(output, (batch_size, sequence_length, self.dimension))
    output = self.output_projection(output) # Apply the output transformation layer

    return output
```

The `SelfAttentionLayer` class encapsulates the attention logic, along with initialization of linear layers and masking strategies, into a self-contained reusable module. The forward method manages the transformations before and after the core attention operation and does so based on the specified masking strategy when constructed. This structure is better suited for complex, modular models, where multiple attention layers may be used, each potentially with different parameters.

Lastly, consider a variant of the function example incorporating multi-head attention:

```python
import hypothetical_tensor_lib as hl

def multi_head_self_attention_function(query, key, value, num_heads, mask=None):
  """
    Computes multi-head self-attention using a functional approach.

      Args:
        query: Tensor of shape (batch_size, sequence_length, dimension).
        key: Tensor of shape (batch_size, sequence_length, dimension).
        value: Tensor of shape (batch_size, sequence_length, dimension).
        num_heads: The number of attention heads.
        mask: Optional boolean mask for attended elements.

    Returns:
      Tensor of shape (batch_size, sequence_length, dimension).
  """
  batch_size, sequence_length, dimension = query.shape
  head_dimension = dimension // num_heads # Individual head dimension

  query = hl.reshape(query, (batch_size, sequence_length, num_heads, head_dimension))
  key   = hl.reshape(key,   (batch_size, sequence_length, num_heads, head_dimension))
  value = hl.reshape(value, (batch_size, sequence_length, num_heads, head_dimension))

  query = hl.transpose(query, 1, 2)
  key   = hl.transpose(key,   1, 2)
  value = hl.transpose(value, 1, 2)


  attention_weights = hl.matmul(query, hl.transpose(key, -1, -2))
  attention_weights = attention_weights / hl.sqrt(head_dimension)

  if mask is not None:
      attention_weights = hl.masked_fill(attention_weights, mask == 0, -1e9)

  attention_weights = hl.softmax(attention_weights, axis=-1)
  output = hl.matmul(attention_weights, value)

  output = hl.transpose(output, 1, 2)
  output = hl.reshape(output, (batch_size, sequence_length, dimension))

  return output
```

This function `multi_head_self_attention_function` encapsulates the multi-head logic, however it still relies on linear transformations, and any masking strategy, to be provided outside the function. The class implementation offers better modularity with all components in a defined structure.

For a deeper understanding of self-attention, resources focused on transformer models offer valuable insight. Texts describing attention mechanisms, specifically within the context of natural language processing, can be highly informative. Additionally, papers detailing the implementation choices in widely used libraries such as TensorFlow or PyTorch can provide practical understanding of the differences in design. Examining the source code of pre-trained transformer models is an excellent way to see these concepts in action. Focus on resources that articulate not only the mathematical background, but also address the implementation considerations.
