---
title: "What are the different shapes of transformers?"
date: "2025-01-30"
id: "what-are-the-different-shapes-of-transformers"
---
The architecture of transformer neural networks is fundamentally defined by the attention mechanism, yet the implementation and arrangement of these mechanisms lead to a diverse landscape of transformer "shapes" tailored to specific tasks and data modalities. My experience developing models for time-series analysis and natural language processing has exposed me to several such variations, each with distinct structural properties.

The most basic form, often referred to as the “vanilla” transformer, is primarily composed of stacked encoder and decoder layers. In a typical encoder, input data (such as token embeddings in NLP) flows through a series of multi-head attention layers followed by feed-forward networks. The multi-head attention allows the model to capture relationships between all elements in the input sequence, weighed by learned attention scores. The feed-forward network, generally a two-layer perceptron with a ReLU activation, introduces non-linearity and allows for feature transformations. The output of an encoder layer becomes the input for the next, allowing for increasingly complex feature abstractions with depth. This is repeated across several encoder layers, which also include residual connections and layer normalization to stabilize training.

The decoder component shares a similar structure but includes an additional attention sub-layer. This extra layer performs attention not on the decoder input sequence itself but rather on the encoded representation provided by the encoder. This "cross-attention" layer is crucial for generating outputs based on the input context. For example, in machine translation, the decoder’s self-attention focuses on the target language being generated, while the cross-attention mechanism ensures that the generated words align with the encoded representation of the source language. The decoder also uses masked self-attention to prevent it from “peeking” at future tokens during training. This masking allows for parallel processing of the output sequence, essential for efficient training, but also enforces an autoregressive structure during inference. In scenarios where output prediction isn't sequential, such as in image processing or certain text analysis tasks, an unmasked decoder may be employed.

This encoder-decoder architecture represents a common shape for sequence-to-sequence problems, such as machine translation, text summarization, and speech recognition. However, numerous deviations exist. Some models utilize only the encoder or the decoder portion of the architecture, adapting to specific objectives.

**Code Example 1: Basic Encoder Layer (Conceptualized)**

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output) # Residual & Layer Norm
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output) # Residual & Layer Norm
        return x

# Example instantiation
d_model = 512
nhead = 8
dim_feedforward = 2048
encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward)
# x = torch.randn(sequence_length, batch_size, d_model) # Input tensor shape
# output = encoder_layer(x) # Output tensor shape
```

This Python example provides a simplified conceptualization of a single encoder layer using PyTorch. It showcases the main components: multi-head attention, feed-forward network, and residual connections with layer normalization. The instantiation demonstrates parameters like the model dimension (`d_model`), the number of attention heads (`nhead`), and the dimension of the feed-forward network (`dim_feedforward`). The forward method represents the flow of data through the layer, which can be repeatedly stacked to form the complete encoder. Notably, this only implements a single encoder layer and not the entire transformer model.

**Code Example 2: Transformer Encoder-Only (Conceptualized)**

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, vocab_size, max_seq_len):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x

class PositionalEncoding(nn.Module): # Simplified implementation for demonstration
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0),:]

# Example instantiation
vocab_size = 10000
max_seq_len = 500
num_layers = 6
d_model = 512
nhead = 8
dim_feedforward = 2048
encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, vocab_size, max_seq_len)
# input_sequence = torch.randint(0, vocab_size, (sequence_length, batch_size)) # Input sequence
# output = encoder(input_sequence) # Output tensor shape

```

This second example shows a conceptual implementation of a transformer encoder-only architecture, commonly used in tasks like text classification or sentence embeddings. It includes the embedding layer to map vocabulary indices into a dense vector space, and crucially, positional encoding, essential for capturing sequence information, since attention is permutation-invariant. Multiple encoder layers are stacked within this encoder, processing the data iteratively. Layer normalization is added before the final output. The code illustrates that this structure takes an input tensor of token indices, embeds it, adds positional encodings, feeds through the encoder layers and outputs the processed representation.

**Code Example 3: Decoder-Only (Conceptualized)**

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + cross_attn_output)
        ff_output = self.feedforward(x)
        x = self.norm3(x + ff_output)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, vocab_size, max_seq_len):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size) # Linear to map to vocab
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, mask)
        x = self.final_norm(x)
        x = self.linear(x)
        return x

# Example instantiation
vocab_size = 10000
max_seq_len = 500
num_layers = 6
d_model = 512
nhead = 8
dim_feedforward = 2048
decoder = TransformerDecoder(d_model, nhead, num_layers, dim_feedforward, vocab_size, max_seq_len)
# decoder_input = torch.randint(0, vocab_size, (sequence_length, batch_size)) # Decoder input sequence
# encoder_output = torch.randn(sequence_length, batch_size, d_model) # Assume from an encoder
# mask =  torch.triu(torch.ones(sequence_length,sequence_length),diagonal=1).bool() # Mask for auto-regressive
# output = decoder(decoder_input, encoder_output, mask) # Output
```

This third example demonstrates a conceptual decoder-only implementation, highlighting masked self-attention and cross-attention. The `DecoderLayer` includes a self-attention layer, a cross-attention layer (taking output from the encoder), and the feedforward network. The `TransformerDecoder` component embeddings the inputs, applies positional encodings, passes them through decoder layers, and applies a linear layer to obtain logits. The `forward` method demonstrates how the decoder is used in a sequence to sequence setup, and the inclusion of `mask` shows how masking is used.

Beyond these core shapes, there exists a wide range of architectures that modify or specialize transformer blocks:

*   **Vision Transformers (ViT):** ViTs apply the standard transformer encoder to image patches, treating them as analogous to text tokens. The patch embeddings are fed into the model and classification is done on a special CLS token's output.
*   **Longformer/BigBird/Reformer:** These models introduce modifications to the attention mechanism to handle much longer input sequences, reducing the quadratic computational complexity of standard attention.
*   **Performer:** Based on kernel methods, Performer can handle very long sequences while maintaining good performance and avoiding the need for quadratic attention.
*   **Hybrid Transformers:** Architectures that combine transformers with other network types, such as convolutional neural networks, to utilize the strengths of each. An example would be combining CNNs to extract low-level visual features, while transformers handle high-level dependencies.

For continued exploration, I recommend studying the original "Attention is All You Need" paper. Additionally, texts specializing in sequence modeling and natural language processing detail the use cases and implementations of various transformer architectures, and open-source repositories, such as those on Github, offer comprehensive implementations for experimentation.
