---
title: "Must target and source sequences have the same feature dimensions in a PyTorch Transformer?"
date: "2025-01-30"
id: "must-target-and-source-sequences-have-the-same"
---
In my experience implementing sequence-to-sequence models, particularly using PyTorch's Transformer, it's crucial to understand that while the *embedding* dimensions of the source and target sequences *must* be identical for a standard Transformer architecture, the overall *feature sequence lengths* can, and often will, differ. This distinction is paramount, as it governs how data flows through the network and prevents common errors related to tensor shape mismatches. The confusion often arises because both source and target sequences need to pass through embedding layers that project them into the same dimensional space, thus leading some to believe the original sequences themselves should have the same lengths. This is not the case.

The core reason for the shared embedding dimensionality lies within the Transformer's attention mechanism. Both the encoder and decoder components of the Transformer rely on calculating attention scores between queries, keys, and values. These three entities must have compatible shapes for matrix multiplication. The embedding layer transforms both source and target token indices (or other initial representations) into dense vectors of the same size, which then act as the keys, queries, and values. If these vectors had differing dimensions, the attention calculations would become mathematically undefined. However, the *number* of tokens, and thus the sequence length of the source and target, can be, and indeed should be, different in most machine translation or text summarization applications.

I will illustrate this with specific code examples and explain the nuances of each step.

**Example 1: Initial Setup and Embedding Layer**

Let us consider a simplified scenario where our vocabulary size for both source and target is 1000, and we aim for an embedding dimensionality of 512. The source sequence might be a sentence with 20 words, and the target sequence could be its translation consisting of 25 words.

```python
import torch
import torch.nn as nn

# Define hyperparameters
vocab_size = 1000
embedding_dim = 512
source_seq_len = 20
target_seq_len = 25
batch_size = 32

# Create embedding layers with the same output dimensionality.
source_embedding = nn.Embedding(vocab_size, embedding_dim)
target_embedding = nn.Embedding(vocab_size, embedding_dim)

# Create dummy input tensors with different sequence lengths
source_indices = torch.randint(0, vocab_size, (batch_size, source_seq_len)) # shape: (batch_size, source_seq_len)
target_indices = torch.randint(0, vocab_size, (batch_size, target_seq_len)) # shape: (batch_size, target_seq_len)

# Embed the input
source_embedded = source_embedding(source_indices)  # shape: (batch_size, source_seq_len, embedding_dim)
target_embedded = target_embedding(target_indices)  # shape: (batch_size, target_seq_len, embedding_dim)

print(f"Shape of embedded source: {source_embedded.shape}")
print(f"Shape of embedded target: {target_embedded.shape}")

```
In this code, both `source_embedding` and `target_embedding` have an output dimension of 512, fulfilling the requirement for compatible embedding spaces. Critically, note that the `source_indices` and `target_indices` tensors have different sequence length dimensions – 20 and 25 respectively. Embedding layers transform sequences into a dense vector representation without altering the batch size or sequence length dimensions.

**Example 2: Transformer Encoder and Decoder Forward Pass**

Following the embedding layers, the sequences pass through the encoder and decoder, which implement attention computations. This is where the dimensional compatibility matters most.

```python
import torch
import torch.nn as nn
from torch.nn import Transformer

# Define hyperparameters (reusing those from Example 1)
vocab_size = 1000
embedding_dim = 512
source_seq_len = 20
target_seq_len = 25
batch_size = 32
num_heads = 8
num_layers = 6

# Create embedding layers (reusing from Example 1)
source_embedding = nn.Embedding(vocab_size, embedding_dim)
target_embedding = nn.Embedding(vocab_size, embedding_dim)

# Create dummy input tensors (reusing from Example 1)
source_indices = torch.randint(0, vocab_size, (batch_size, source_seq_len))
target_indices = torch.randint(0, vocab_size, (batch_size, target_seq_len))

# Embed the input
source_embedded = source_embedding(source_indices)
target_embedded = target_embedding(target_indices)


# Create a basic Transformer
transformer = Transformer(
    d_model=embedding_dim,
    nhead=num_heads,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers
)

# Encoder forward pass: source sequence goes through encoder layers.
encoded_source = transformer.encoder(source_embedded) # shape: (batch_size, source_seq_len, embedding_dim)

# Decoder forward pass. target sequence and encoder outputs are passed.
# We use a target mask to enforce autoregressive behavior during training.
target_mask = transformer.generate_square_subsequent_mask(target_seq_len)
decoded_output = transformer.decoder(target_embedded, encoded_source, tgt_mask=target_mask) # shape: (batch_size, target_seq_len, embedding_dim)

print(f"Shape of encoded source output: {encoded_source.shape}")
print(f"Shape of decoded target output: {decoded_output.shape}")


```

Observe that the output of the encoder `encoded_source` retains the source sequence length (20 in this example), while the decoder's output `decoded_output` has the target sequence length (25). Importantly, the `embedding_dim` stays constant throughout the forward passes. The `Transformer` class handles attention score calculations internally, without requiring external reshaping, due to consistent embedding dimensions. The critical part is how the encoder output is used as part of cross attention mechanism in the decoder, which needs to be compatible with the decoder's target sequence features. The source sequence itself can be any length.

**Example 3: Addressing Length Discrepancies with Masking**

While sequence lengths can differ, there are situations where padding is necessary, such as when batching sequences with variable lengths, leading to the need for masks. This ensures that padding tokens are ignored during attention calculations.

```python
import torch
import torch.nn as nn
from torch.nn import Transformer

# Define hyperparameters (reusing from previous examples)
vocab_size = 1000
embedding_dim = 512
source_seq_len = 20
target_seq_len = 25
batch_size = 32
num_heads = 8
num_layers = 6

# Create embedding layers (reusing from previous examples)
source_embedding = nn.Embedding(vocab_size, embedding_dim)
target_embedding = nn.Embedding(vocab_size, embedding_dim)

# Create dummy input tensors (reusing from previous examples, with some modification)
source_seq_lengths = torch.randint(10, source_seq_len+1, (batch_size,)) # Varying lengths
target_seq_lengths = torch.randint(15, target_seq_len+1, (batch_size,)) # Varying lengths

source_indices = torch.zeros((batch_size, source_seq_len), dtype=torch.long)
target_indices = torch.zeros((batch_size, target_seq_len), dtype=torch.long)


for i in range(batch_size):
  source_indices[i, :source_seq_lengths[i]] = torch.randint(0, vocab_size, (source_seq_lengths[i],))
  target_indices[i, :target_seq_lengths[i]] = torch.randint(0, vocab_size, (target_seq_lengths[i],))


# Embed the input
source_embedded = source_embedding(source_indices)
target_embedded = target_embedding(target_indices)

# Create masks
source_mask = (source_indices != 0).float().unsqueeze(1).unsqueeze(2) # Shape: (batch_size, 1, 1, source_seq_len)
target_mask = (target_indices != 0).float().unsqueeze(1).unsqueeze(2) # Shape: (batch_size, 1, 1, target_seq_len)


# Create a basic Transformer (reusing from Example 2)
transformer = Transformer(
    d_model=embedding_dim,
    nhead=num_heads,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers
)

# Encoder forward pass
encoded_source = transformer.encoder(source_embedded, src_key_padding_mask=source_mask.squeeze(1).squeeze(1)==0 )

# Decoder forward pass
square_target_mask = transformer.generate_square_subsequent_mask(target_seq_len)
decoded_output = transformer.decoder(target_embedded, encoded_source, tgt_mask=square_target_mask, tgt_key_padding_mask=target_mask.squeeze(1).squeeze(1)==0)


print(f"Shape of masked encoded source output: {encoded_source.shape}")
print(f"Shape of masked decoded target output: {decoded_output.shape}")
```

In this third example, we simulate sequences of varying lengths within a batch.  A mask is created that is ‘1’ for valid sequence tokens, and ‘0’ for padded tokens. This mask is then used within the `transformer.encoder` and `transformer.decoder` calls using the keyword arguments, ensuring padded tokens are excluded from the attention scores. The padding does not affect the embedding dimension, it only ensures the Transformer processes and learns the correct sequence information.

To summarize, the embedding dimension must match for source and target sequences within a PyTorch Transformer. The source and target sequences can have different lengths, and masks are crucial when batching data that has variable sequence lengths.

For further study, I recommend reviewing the original Transformer paper ("Attention is All You Need") along with practical examples in the PyTorch documentation on `torch.nn.Transformer`.  Also, research the use of masking techniques in attention mechanisms.  Books on deep learning that discuss sequence modeling and attention would also prove helpful, particularly those focusing on natural language processing.
