---
title: "Why are weight matrices shared across embedding layers in the 'Attention is All You Need' paper?"
date: "2025-01-30"
id: "why-are-weight-matrices-shared-across-embedding-layers"
---
Shared weight matrices between the input and output embedding layers in the "Attention is All You Need" architecture, as introduced by Vaswani et al., are primarily driven by a desire for parameter efficiency and a simplified model architecture that benefits from a close correspondence between token representations in both encoding and decoding phases. This decision, while seemingly straightforward, is grounded in an understanding of the underlying linguistic structures and the nature of embedding spaces.

In my experience building several sequence-to-sequence models, I’ve observed firsthand the impact of parameter count on both training time and model performance. Using distinct weight matrices for input and output embeddings substantially increases the number of trainable parameters, particularly when dealing with large vocabulary sizes. By sharing these matrices, we reduce this parameter burden, thereby speeding up training and potentially mitigating overfitting. The logic behind this sharing is not arbitrary; it’s predicated on the assumption that the semantic space for input tokens and output tokens (within the same vocabulary) can be meaningfully represented by the same embedding space. The learned embeddings, although used in different contexts (encoder input and decoder output), can be argued to represent similar underlying semantic and syntactic structures.

The core rationale hinges on viewing token representations as points within a continuous high-dimensional space. Ideally, words or sub-words with similar meanings or functions should cluster together in this space, regardless of whether they’re being inputted to the encoder or predicted by the decoder. Sharing the weight matrices facilitates this by forcing both encoder and decoder to operate using the same ‘coordinate system’ in this space. This constraint can also help establish a stronger connection between the representations, allowing the decoder to better interpret the encoded information. Consider it a unified semantic dictionary, accessed by both the encoding and decoding process. This reduces the burden of learning two disparate mapping systems, potentially improving generalization.

The paper also introduces a scaling factor of sqrt(d_model) to the output embeddings within the model. This scaling compensates for the smaller magnitudes of variance arising from the weight sharing. Without this scaling, the encoder's attention weights may be significantly different and produce suboptimal results.

Let’s examine this with some conceptual code examples. These examples are simplified and for illustration only, they do not represent the actual implementation of Attention is All You Need. The assumed environment is Python with a numerical computing library (like NumPy).

**Example 1: No Weight Sharing**

Here's how the embedding lookup would conceptually work *without* shared weight matrices:

```python
import numpy as np

vocab_size = 1000
embedding_dim = 512

# Separate weight matrices for input and output
input_embedding_weights = np.random.randn(vocab_size, embedding_dim)
output_embedding_weights = np.random.randn(vocab_size, embedding_dim)

def embed_input(input_tokens):
  """Embeds input tokens using the input embedding weights."""
  return np.array([input_embedding_weights[token] for token in input_tokens])

def embed_output(output_tokens):
  """Embeds output tokens using the output embedding weights."""
  return np.array([output_embedding_weights[token] for token in output_tokens])

# Example
input_sequence = [10, 250, 78, 99]
output_sequence = [1, 45, 67, 12]

input_embeddings = embed_input(input_sequence)
output_embeddings = embed_output(output_sequence)

print(f"Input embeddings shape: {input_embeddings.shape}") # Output: (4, 512)
print(f"Output embeddings shape: {output_embeddings.shape}") # Output: (4, 512)
```

This example highlights that we maintain *two* distinct weight matrices, `input_embedding_weights` and `output_embedding_weights`, each requiring its own set of parameters to be learned during training. The function `embed_input` uses one and `embed_output` uses the other for lookup, resulting in two separate embedding spaces for our tokens.

**Example 2: Weight Sharing**

Now, let's look at an implementation with *shared* weight matrices:

```python
import numpy as np

vocab_size = 1000
embedding_dim = 512

# Shared weight matrix for both input and output
shared_embedding_weights = np.random.randn(vocab_size, embedding_dim)

def embed_shared(tokens):
    """Embeds tokens using the shared embedding weights."""
    return np.array([shared_embedding_weights[token] for token in tokens])

# Example
input_sequence = [10, 250, 78, 99]
output_sequence = [1, 45, 67, 12]

input_embeddings = embed_shared(input_sequence)
output_embeddings = embed_shared(output_sequence)


print(f"Input embeddings shape: {input_embeddings.shape}") # Output: (4, 512)
print(f"Output embeddings shape: {output_embeddings.shape}") # Output: (4, 512)
```

Here, only `shared_embedding_weights` is used, reducing the number of learned parameters by half for the embedding layers. Both `input_embeddings` and `output_embeddings` are generated using the same matrix, ensuring the same embedding space is applied for both input and output sequences.

**Example 3: Scaled Embedding Output (Conceptual)**

Lastly, the output embeddings, before going to the final linear transformation, are scaled by sqrt(d_model), where d_model is the embedding dimension.

```python
import numpy as np
import math

vocab_size = 1000
embedding_dim = 512

# Shared weight matrix for both input and output
shared_embedding_weights = np.random.randn(vocab_size, embedding_dim)

def embed_shared_scaled(tokens):
    """Embeds tokens using the shared embedding weights and scales the output."""
    embeddings = np.array([shared_embedding_weights[token] for token in tokens])
    return embeddings * math.sqrt(embedding_dim)

def linear_transformation(embeddings):
  #Conceptual linear transformation
  weights = np.random.randn(embedding_dim, vocab_size)
  return np.dot(embeddings, weights)
    

# Example
output_sequence = [1, 45, 67, 12]


output_embeddings_scaled = embed_shared_scaled(output_sequence)

linear_output = linear_transformation(output_embeddings_scaled)

print(f"Scaled output embeddings shape: {output_embeddings_scaled.shape}")  # Output: (4, 512)
print(f"Final linear output shape: {linear_output.shape}")  # Output: (4, 1000)
```

The `embed_shared_scaled` function is modified to include the scaling factor `math.sqrt(embedding_dim)`. This operation is applied before the decoder output embeddings go through the final linear transformation, as demonstrated with the simple conceptual linear transformation, and helps to maintain the appropriate scales of attention weights. This isn't part of the embedding look up, but a crucial step after it, that is mentioned with the discussion of weight sharing.

For anyone seeking a deeper understanding, I recommend consulting the original "Attention is All You Need" paper for the detailed architecture diagram and explanations. The "Transformer: A Novel Neural Network Architecture" blog series (published by Lilian Weng) offers a great explanation of the technical aspects and motivations behind the model, focusing on the attention mechanism and overall architecture. The book "Natural Language Processing with Transformers" provides an in-depth look at the practical usage and implementation of transformers, including details of weight sharing. The open-source documentation of the Hugging Face Transformers library, along with source code exploration, is an excellent way to dive into specific implementation details. These resources collectively provide a solid foundation for understanding the nuances of the transformer architecture and its design choices, such as weight sharing.
