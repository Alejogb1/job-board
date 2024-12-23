---
title: "What are the shapes of transformers?"
date: "2024-12-23"
id: "what-are-the-shapes-of-transformers"
---

Let's consider for a moment, not necessarily the literal shapes of the physical hardware of transformer models, but more precisely, the conceptual shapes that define their internal workings. We're talking about the architectures and configurations which give these models their specific capabilities, and as someone who's spent a considerable amount of time optimizing and debugging transformer networks, I can say firsthand that understanding these 'shapes' is critical for success.

Essentially, when we say "shapes of transformers," we're often referring to the different arrangements and modifications of the core transformer architecture—the one introduced in the "Attention is All You Need" paper by Vaswani et al. This foundational structure, featuring encoder and decoder blocks, has been adapted to tackle diverse problems, and these variations often exhibit unique structural ‘shapes.’

The simplest “shape,” if we can even call it that, is the standard encoder-decoder transformer. Here, data flows in one direction during the encoding phase and then into the decoder for generation. The encoder, typically composed of several identical layers, ingests the input sequence (say a sentence in natural language). Each layer performs multi-head self-attention, where the input attends to itself in multiple subspaces, and then applies a feed-forward network. This process transforms the input representation into a context-aware vector representation. The decoder, similarly stacked, also incorporates attention mechanisms—both self-attention (within the decoder) and cross-attention (attention to the encoded representation produced by the encoder). This cross-attention mechanism is key for tasks like translation, as it allows the decoder to condition its output on the input representation. This is essentially a shape of two interconnected towers, an encoder and a decoder, with the information flowing from one to the other.

A crucial element in almost all transformer shapes is the attention mechanism. It's more than just an element; it's the core principle behind transformers' ability to capture long-range dependencies. Attention calculates weights based on the relationships between different tokens in the input. These weights then determine which parts of the input are most relevant for processing a given token. This effectively allows the model to focus on relevant parts of the input, and this mechanism often results in the ‘shapes’ of the interaction between inputs. You could think of it as a dynamic network, its connections evolving based on the content of the input.

Now, let's look at some concrete examples. First, the standard encoder structure mentioned previously. This particular configuration, often referred to as "the encoder stack," involves several layers. Here’s a snippet, using a hypothetical, simplified pytorch example, to get the idea.

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
      attn_output, _ = self.self_attn(x, x, x) #Self-attention
      x = self.norm1(x + attn_output)
      ff_output = self.feedforward(x)
      x = self.norm2(x + ff_output)
      return x

class Encoder(nn.Module):
  def __init__(self, num_layers, d_model, nhead, dim_feedforward):
    super(Encoder, self).__init__()
    self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

  def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x

# Example usage
input_size = 10
d_model = 512
nhead = 8
dim_feedforward = 2048
num_layers = 6
encoder = Encoder(num_layers, d_model, nhead, dim_feedforward)
input_tensor = torch.randn(input_size, 1, d_model)
output = encoder(input_tensor)
print(output.shape) # Output: torch.Size([10, 1, 512])
```
This code outlines a basic encoder structure. You can see the repeating ‘shape’ of layers that progressively process the input. The output of each layer feeds into the next, creating a stack.

Then there's the decoder, which has a different "shape," especially when combined with the encoder in sequence-to-sequence models. The decoder also incorporates masked self-attention to prevent peeking into future tokens, and it also performs cross-attention with the encoded representation. Here’s a snippet that focuses just on the cross-attention aspect within a hypothetical decoder layer:

```python
import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, d_model, nhead)
        self.cross_attn = nn.MultiheadAttention(d_model, d_model, nhead)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask) # Self-attention with masking
        x = self.norm1(x + attn_output)
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output) # Cross attention
        x = self.norm2(x + cross_attn_output)
        ff_output = self.feedforward(x)
        x = self.norm3(x + ff_output)
        return x

#Example usage
d_model = 512
nhead = 8
dim_feedforward = 2048

decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward)
input_size = 10
batch_size = 1
encoder_output_size = 10

decoder_input = torch.randn(input_size, batch_size, d_model)
encoder_output = torch.randn(encoder_output_size, batch_size, d_model)
output = decoder_layer(decoder_input, encoder_output)

print(output.shape) #Output torch.Size([10, 1, 512])

```
This snippet emphasizes the cross-attention component. The decoder layer attends to both its own representation and the encoder’s output. Masking during self-attention is crucial to maintain proper autoregressive behavior.

Finally, there are also configurations such as encoder-only models, which we see in models like BERT. These are characterized by the absence of a decoder. Instead, a task-specific layer is added on top of the encoder output, such as a classification head. These models typically utilize a single stack of encoder layers and are trained to predict masked tokens within the input. This gives us another distinct "shape":

```python
import torch
import torch.nn as nn

class BertEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
      super(BertEncoderLayer, self).__init__()
      self.self_attn = nn.MultiheadAttention(d_model, nhead)
      self.feedforward = nn.Sequential(
          nn.Linear(d_model, dim_feedforward),
          nn.GELU(),
          nn.Linear(dim_feedforward, d_model)
      )
      self.norm1 = nn.LayerNorm(d_model)
      self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
      attn_output, _ = self.self_attn(x, x, x) #Self-attention
      x = self.norm1(x + attn_output)
      ff_output = self.feedforward(x)
      x = self.norm2(x + ff_output)
      return x

class BertEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward):
      super(BertEncoder, self).__init__()
      self.layers = nn.ModuleList([BertEncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

    def forward(self, x):
      for layer in self.layers:
        x = layer(x)
      return x


#Example Usage
input_size = 10
d_model = 512
nhead = 8
dim_feedforward = 2048
num_layers = 6

bert_encoder = BertEncoder(num_layers, d_model, nhead, dim_feedforward)
input_tensor = torch.randn(input_size, 1, d_model)
output = bert_encoder(input_tensor)

print(output.shape) # Output torch.Size([10, 1, 512])
```

This modified encoder demonstrates the basic structure of encoder-only transformers, omitting the decoder and often incorporating GELU activation functions. It highlights a shape focused on understanding context without direct generative tasks.

It's also worth briefly mentioning variations like "transformer-xl" which employs recurrence to deal with long sequences, creating a shape that processes data sequentially while maintaining contextual memory across segments. Or even the "efficient transformers," which have tried to use different attention mechanisms to deal with the computation bottleneck that is self-attention by reconfiguring the layers for sparse attention. These too change the overall 'shape' we are talking about.

To gain a deeper understanding of these different 'shapes,' I’d highly recommend reviewing “Attention is All You Need,” (Vaswani et al., 2017). For a more comprehensive look at transformer models and their numerous variations, “Natural Language Processing with Transformers” by Tunstall, von Werra, and Wolf, is invaluable. And for exploring the theoretical underpinnings and mathematics involved, “Deep Learning” by Goodfellow, Bengio, and Courville, will provide a solid foundation.

In conclusion, the “shapes” of transformers are defined not only by their encoder and decoder compositions, but also by the nuanced arrangement and the function of self and cross attention. These different shapes allow for a wide range of applications, and understanding these configurations is essential for anyone working with these powerful architectures.
