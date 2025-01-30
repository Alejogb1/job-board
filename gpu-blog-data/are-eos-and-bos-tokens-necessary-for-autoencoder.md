---
title: "Are EOS and BOS tokens necessary for autoencoder transformers?"
date: "2025-01-30"
id: "are-eos-and-bos-tokens-necessary-for-autoencoder"
---
The necessity of End-of-Sentence (EOS) and Beginning-of-Sentence (BOS) tokens for autoencoder transformers hinges entirely on the specific task and model architecture. While not universally required, their absence often leads to degraded performance or an inability to delineate sequence boundaries, particularly for sequence-to-sequence tasks, which many autoencoders implicitly perform. Having spent the last eight years architecting various autoencoding models, I’ve observed a clear pattern in when these tokens are beneficial versus when they’re redundant.

At its core, the purpose of an autoencoder is to reconstruct its input. A vanilla autoencoder, trained on individual words or fixed-size vectors, might not require special tokens because it’s dealing with isolated units. However, when the input consists of variable-length sequences, such as sentences or paragraphs, the model needs a mechanism to understand the start and end of those sequences. This is where EOS and BOS tokens become crucial. They explicitly encode the boundaries of a sequence, allowing the model to learn meaningful representations that take context into account. In the absence of these tokens, the model might struggle to differentiate between a genuine sequence ending and a premature cut-off, especially during the decoding phase, potentially leading to the generation of incomplete or illogical sequences.

The role of these special tokens is most pronounced in sequence-to-sequence tasks. For instance, consider an autoencoder that’s trained to compress text sequences. The encoder maps the input sentence, which includes a BOS token at the start and an EOS token at the end, into a compressed latent space. The decoder, then, takes this compressed representation and attempts to reconstruct the original sequence, again, starting with the BOS token and continuing until the EOS token is generated. Without these boundary markers, the decoder wouldn’t reliably know when to start generating output or when to stop, potentially creating spurious or incomplete output sequences.

The presence of EOS tokens also influences training convergence. The loss function often incorporates the EOS prediction as part of the overall sequence reconstruction. If the model predicts an incorrect EOS location, it receives feedback that helps it learn the underlying sequence patterns better. The BOS token often aids in starting the attention mechanism correctly, specifically in Transformer architectures where the first input token sets the context for attending to the rest of the sequence.

The necessity further depends on the architecture. A simple feed-forward autoencoder working on fixed-length numerical vectors might be fine without EOS or BOS tokens. However, as soon as the autoencoder incorporates any form of recurrence (e.g., LSTMs, GRUs) or attention (as in transformers), these tokens become substantially more beneficial. The recurrent networks need to understand the sequence flow and when to reset their internal states. Attention mechanisms, like those found in transformers, need an initial context vector to initiate their operation, which the BOS token facilitates.

Let's examine three distinct scenarios with code snippets to clarify the point:

**Scenario 1: Vanilla Autoencoder with Fixed-Length Vectors (No Tokens)**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return decoded

input_size = 10
hidden_size = 5
model = SimpleAutoencoder(input_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Sample input (batch size of 3)
input_tensor = torch.randn(3, input_size)

#Training loop (simplified)
optimizer.zero_grad()
output = model(input_tensor)
loss = criterion(output, input_tensor)
loss.backward()
optimizer.step()

print("Loss:", loss.item())
```

*Commentary:* In this simplified example, the input is a fixed-size vector.  There's no concept of a 'sequence', so no BOS or EOS tokens are necessary. The autoencoder directly processes and reconstructs the input vector, with no sequential context. This illustrates a case where these special tokens bring no benefit.

**Scenario 2: Sequence Autoencoder with Recurrent Layer and EOS token (Required)**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SequenceAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(SequenceAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, embedding_dim, batch_first=True)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.encoder(embedded)  # Hidden state becomes the encoded representation
        decoder_input = self.embedding(x)
        decoded, _ = self.decoder(decoder_input, (hidden, torch.zeros_like(hidden)))
        output = self.output_layer(decoded)
        return output

vocab_size = 100
embedding_dim = 20
hidden_size = 40
model = SequenceAutoencoder(vocab_size, embedding_dim, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#Sample sequence including an EOS token (represented by vocab index 99)
input_sequence = torch.tensor([[10, 20, 30, 99], [40, 50, 60, 99]]) #batch of size two
#Training Loop simplified
optimizer.zero_grad()
output = model(input_sequence)
loss = criterion(output.view(-1, vocab_size), input_sequence.view(-1))
loss.backward()
optimizer.step()

print("Loss:", loss.item())
```
*Commentary:* Here, we have an LSTM-based autoencoder designed for variable-length sequences. The input `input_sequence` includes an EOS token (represented as 99).  Without this EOS token, the model would not know when to stop decoding, and loss calculations would be inaccurate. While a BOS was not explicitly fed to the decoder (which is common), the decoder is still dependent on the hidden vector passed from the encoder, which indirectly captures the sequential context, including the EOS token.

**Scenario 3: Transformer Autoencoder with BOS and EOS (Crucial)**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_heads, num_layers):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_size)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.decoder_layers = nn.TransformerDecoderLayer(embedding_dim, num_heads, hidden_size)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, tgt):
       src_embedded = self.pos_encoder(self.embedding(src))
       tgt_embedded = self.pos_encoder(self.embedding(tgt))
       memory = self.encoder(src_embedded)
       decoded_output = self.decoder(tgt_embedded, memory)
       output = self.output_layer(decoded_output)
       return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
      return x + self.pe[:, :x.size(1)]


vocab_size = 100
embedding_dim = 20
hidden_size = 40
num_heads = 2
num_layers = 2
model = TransformerAutoencoder(vocab_size, embedding_dim, hidden_size, num_heads, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Sample input sequence with BOS (index 0) and EOS (index 99)
src_sequence = torch.tensor([[0,10, 20, 30, 99],[0,40, 50, 60,99]]) #batch of size 2
tgt_sequence = torch.tensor([[0,10, 20, 30, 99],[0,40, 50, 60,99]]) #batch of size 2

#Training Loop simplified
optimizer.zero_grad()
output = model(src_sequence,tgt_sequence)
loss = criterion(output.view(-1, vocab_size), tgt_sequence.view(-1))
loss.backward()
optimizer.step()
print("Loss:", loss.item())
```

*Commentary:* This transformer autoencoder explicitly uses both BOS (index 0) and EOS (index 99).  The `src_sequence` is input to the encoder, and the `tgt_sequence` acts as a target for the decoder, similar to a sequence-to-sequence task. This example highlights their critical role for transformers in sequence autoencoding. The BOS token aids in establishing a consistent initial context for the decoder attention, while the EOS helps the model identify the termination of the sequence, particularly important during generative tasks with the decoder.

In conclusion, whether you need EOS and BOS tokens in your autoencoder transformers depends strongly on your specific use case and architecture. While not necessary for all autoencoders, especially those dealing with isolated units, their crucial for models designed for sequence processing, particularly those based on recurrent or transformer based networks.

For resources, I highly recommend researching papers on Transformer architectures including the original "Attention is All You Need" paper, the foundational work on Sequence to Sequence learning with Recurrent Neural Networks, and more contemporary autoencoder architectures designed for sequence data. Examining specific implementations in libraries such as PyTorch and TensorFlow will further clarify practical application details. Textbooks on deep learning also provide useful background and explanations.
