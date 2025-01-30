---
title: "How to use attention masks in multi-head attention layers in Keras/Tensorflow?"
date: "2025-01-30"
id: "how-to-use-attention-masks-in-multi-head-attention"
---
Attention mechanisms, particularly multi-head attention, are crucial for the success of modern transformer-based models.  However, effectively utilizing attention masks within Keras/TensorFlow to control information flow during the attention computation requires a nuanced understanding of the underlying mechanics.  My experience implementing these in large-scale sequence-to-sequence models for natural language processing has highlighted the importance of precise mask application to avoid information leakage and ensure correct model behavior.  This necessitates a deep dive into the tensor manipulation involved.

**1. Clear Explanation**

Multi-head attention operates by computing attention weights between all pairs of input tokens.  This leads to an attention matrix where each element represents the attention weight between a pair of tokens.  However, in many scenarios, we need to selectively prevent the model from attending to certain tokens.  For example, in machine translation, we might want to prevent the decoder from attending to future tokens in the target sequence (to prevent cheating) or in text classification, we might want to mask out padding tokens.  This selective attention is controlled using attention masks.

These masks are typically binary tensors of the same shape as the attention weight matrices.  A value of 1 indicates that the corresponding attention weight is allowed, while a value of 0 effectively nullifies the weight by setting it to negative infinity (in practice, a very large negative value to avoid numerical instability) before the softmax operation.  This ensures that the model's attention is focused only on the permitted tokens.  The crucial step lies in applying this mask *before* the softmax normalization of the attention weights. Applying it afterwards would not prevent the model from seeing the information - the softmax will still redistribute probability mass.

Crucially, the shape of the mask needs to be meticulously aligned with the attention weights to ensure correct application.  A common source of errors arises from mismatched dimensions, leading to incorrect masking and potentially flawed model predictions. The masking operation should be performed element-wise, multiplying the attention weights with the mask.  The choice between masking with zeros and negative infinity depends on the specific softmax implementation and numerical stability considerations, with the latter often preferred for better gradient flow during training.


**2. Code Examples with Commentary**

**Example 1: Padding Mask**

This example demonstrates creating and applying a padding mask to handle sequences of varying lengths.  This is a common scenario where we need to ignore padding tokens added to make sequences the same length.

```python
import tensorflow as tf

def create_padding_mask(seq):
  """Creates a mask for padding tokens."""
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32) #Identify padding tokens (assuming 0 represents padding)
  return seq[:, tf.newaxis, tf.newaxis, :] # Reshape for broadcasting with attention weights

# Example usage:
batch_size = 2
seq_len = 5
input_seq = tf.constant([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=tf.int32)
padding_mask = create_padding_mask(input_seq)

# ... (Multi-head attention calculation) ...

attention_weights = tf.random.normal((batch_size, num_heads, seq_len, seq_len)) # Example attention weights

masked_attention_weights = attention_weights * padding_mask  + (-1e9) * (1 - padding_mask) #apply mask

# ... (Softmax and further attention calculations) ...
```

This code first identifies padding tokens (assuming 0 represents padding). Then, it reshapes the resulting binary mask to be compatible with the attention weight tensor, ensuring correct broadcasting during the element-wise multiplication.  The added term `(-1e9) * (1 - padding_mask)` effectively sets the weights corresponding to padding tokens to negative infinity.


**Example 2: Look-ahead Mask**

This example illustrates creating and applying a look-ahead mask in a decoder to prevent attending to future tokens during training.  This is vital for autoregressive models.

```python
import tensorflow as tf
import numpy as np

def create_look_ahead_mask(size):
  """Creates a look-ahead mask."""
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask

# Example usage
seq_len = 5
look_ahead_mask = create_look_ahead_mask(seq_len)

# ... (Multi-head attention calculation) ...

attention_weights = tf.random.normal((1, num_heads, seq_len, seq_len)) # Example attention weights for a single sequence

masked_attention_weights = attention_weights + look_ahead_mask * (-1e9) #apply mask

# ... (Softmax and further attention calculations) ...
```

Here, `tf.linalg.band_part` creates a lower triangular matrix.  Subtracting this from an all-ones matrix produces the look-ahead mask, where the upper triangle is 0 (allowing attention) and the lower triangle is 1 (masking).  This effectively prevents the model from "seeing" future tokens.


**Example 3: Combining Masks**

In many situations, you'll need to combine padding and look-ahead masks.

```python
import tensorflow as tf

# Assume padding_mask and look_ahead_mask are already created as in examples 1 and 2

combined_mask = tf.maximum(padding_mask, look_ahead_mask) #Combine masks

attention_weights = tf.random.normal((batch_size, num_heads, seq_len, seq_len))

masked_attention_weights = attention_weights * combined_mask + (-1e9) * (1 - combined_mask)


# ... (Softmax and further attention calculations) ...

```

This example shows how to combine the two masks using element-wise maximum.  This ensures that if either mask indicates a token should be masked, the combined mask will also indicate masking for that token.


**3. Resource Recommendations**

The official TensorFlow documentation provides comprehensive information on tensor manipulation and attention mechanisms.  Explore the relevant sections on `tf.keras.layers.MultiHeadAttention` and tensor operations.  Furthermore, a deep understanding of linear algebra, particularly matrix operations, will be beneficial.  Reviewing materials on matrix multiplication and broadcasting will solidify your grasp of the underlying mathematical principles involved in attention mask applications.   Finally, studying the source code of established transformer implementations can provide valuable insights into best practices and common pitfalls.
