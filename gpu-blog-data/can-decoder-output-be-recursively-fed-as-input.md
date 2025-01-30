---
title: "Can decoder output be recursively fed as input?"
date: "2025-01-30"
id: "can-decoder-output-be-recursively-fed-as-input"
---
In sequence-to-sequence models, specifically those utilizing an encoder-decoder architecture, the practice of feeding the decoder’s output back as input, often called autoregressive decoding or recurrent decoding, is fundamental. This feedback loop allows the decoder to generate outputs one element at a time, conditioned on the previously generated elements, which provides crucial context for each subsequent prediction. If we consider that the encoder typically processes the full input sequence in parallel, and then outputs a context vector, the decoder’s task is to use that context vector, along with its prior output history, to generate an output sequence. This process is inherently sequential and necessitates the recursive use of past predictions.

Let me illustrate this concept by drawing from my work on a machine translation project. In our translation system, we used an encoder-decoder model with recurrent neural networks (RNNs). The encoder ingested a source sentence, such as "The cat sat on the mat," and produced a context vector summarizing its semantic content. The decoder then attempted to generate the corresponding target sentence, “Le chat était assis sur le tapis.” This was not a single, monolithic prediction; rather, it unfolded step-by-step, word-by-word. At the very beginning, the decoder received the context vector, combined it with a special start-of-sequence token (e.g., `<start>`), and generated the first target word, “Le.” This output “Le,” was then fed back into the decoder for the next step, where the model, having now seen the context vector and “Le,” generated "chat". This iterative process continued until the model generated the end-of-sequence token, `<end>`, thereby completing the target sequence. The recursive use of prior predictions is precisely what defines this type of decoding process. Without this recursion, the decoder would be unable to effectively establish a sequential dependence between target words.

Now, let us examine this process in more concrete terms with some simplified code snippets utilizing the Python library PyTorch.

**Example 1: Basic Autoregressive Loop**

This example showcases the fundamental structure of the autoregressive loop. Assume we have a basic RNN decoder, represented by `decoder`, and a context vector `context`. The goal here is to show how, after obtaining a prediction, that output is recycled for the subsequent prediction.

```python
import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(SimpleDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden_state, context):
        embedded = self.embedding(input_token)
        output, hidden = self.rnn(embedded.unsqueeze(0), hidden_state)
        output = self.fc(output.squeeze(0))
        return output, hidden

embedding_dim = 128
hidden_dim = 256
output_dim = 100
decoder = SimpleDecoder(embedding_dim, hidden_dim, output_dim)
context = torch.randn(1, hidden_dim) # Simulating the context from the encoder
start_token = torch.tensor([0]) # The <start> token
max_length = 10 # Maximum length of the output sequence
hidden = context.unsqueeze(0) # Initialize hidden state with context

decoded_tokens = []
current_token = start_token

for _ in range(max_length):
    output, hidden = decoder(current_token, hidden, context)
    predicted_token = torch.argmax(output, dim=1)
    decoded_tokens.append(predicted_token.item())
    current_token = predicted_token  # RECURSIVE FEEDBACK

    if predicted_token.item() == 1 : # Simulating <end> token
        break;
print(decoded_tokens)
```

This snippet highlights the core mechanics of feeding decoder outputs back as inputs. The `current_token` is initialized to the `<start>` token, and after each step, it’s updated with the predicted token. This new `current_token` is then fed into the decoder on the subsequent step. Note that the actual prediction process involves passing the output from the linear layer, which provides logits, through an argmax operation to select the most likely next token.

**Example 2: Conditioning on Both Context and Previous Output**

While the previous example shows a basic recursion, it’s crucial to also emphasize how past output influences the next step *along with* the context vector. This version includes both.

```python
import torch
import torch.nn as nn

class ConditionalDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, context_dim):
        super(ConditionalDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + context_dim, hidden_dim) # Context is also input to the RNN
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden_state, context):
        embedded = self.embedding(input_token)
        combined_input = torch.cat((embedded, context), dim=1).unsqueeze(0)
        output, hidden = self.rnn(combined_input, hidden_state)
        output = self.fc(output.squeeze(0))
        return output, hidden

embedding_dim = 128
hidden_dim = 256
output_dim = 100
context_dim = hidden_dim # Matching dimensions
decoder = ConditionalDecoder(embedding_dim, hidden_dim, output_dim, context_dim)
context = torch.randn(1, context_dim)
start_token = torch.tensor([0])
max_length = 10
hidden = torch.randn(1, 1, hidden_dim) # Initializing hidden state.
decoded_tokens = []
current_token = start_token


for _ in range(max_length):
    output, hidden = decoder(current_token, hidden, context)
    predicted_token = torch.argmax(output, dim=1)
    decoded_tokens.append(predicted_token.item())
    current_token = predicted_token

    if predicted_token.item() == 1 :
       break;
print(decoded_tokens)
```

Here, the primary change is in the `ConditionalDecoder` class. The GRU unit now receives the embedded previous token *concatenated* with the context vector, making the prediction at each step conditional on *both* of these pieces of information. This models the dependency more accurately than the first example.

**Example 3: Incorporating Attention**

A significant refinement to this process is the inclusion of attention mechanisms. These mechanisms allow the decoder to weigh the importance of different parts of the encoded source sequence when generating each word in the target sequence. This example introduces a highly simplified attention mechanism in the decoder.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, context_dim):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.attn = nn.Linear(hidden_dim + context_dim, 1)  # Simple attention mechanism
        self.rnn = nn.GRU(embedding_dim + context_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden_state, context_vector, encoder_outputs):
        embedded = self.embedding(input_token)
        combined_input = torch.cat((hidden_state.squeeze(0), context_vector), dim=1)
        attn_weights = F.softmax(self.attn(combined_input).transpose(0,1), dim = 2) # Transpose needed to use bmm
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1)).transpose(0, 1)
        combined_rnn_input = torch.cat((embedded, context.squeeze(0)), dim=1).unsqueeze(0)
        output, hidden = self.rnn(combined_rnn_input, hidden_state)
        output = self.fc(output.squeeze(0))
        return output, hidden

embedding_dim = 128
hidden_dim = 256
output_dim = 100
context_dim = hidden_dim
decoder = AttentionDecoder(embedding_dim, hidden_dim, output_dim, context_dim)
encoder_outputs = torch.randn(5, 1, hidden_dim) # Simulating outputs of an encoder
context = torch.randn(1, context_dim)
start_token = torch.tensor([0])
max_length = 10
hidden = torch.randn(1, 1, hidden_dim)
decoded_tokens = []
current_token = start_token


for _ in range(max_length):
    output, hidden = decoder(current_token, hidden, context, encoder_outputs)
    predicted_token = torch.argmax(output, dim=1)
    decoded_tokens.append(predicted_token.item())
    current_token = predicted_token

    if predicted_token.item() == 1:
        break;
print(decoded_tokens)
```

The `AttentionDecoder` introduces a rudimentary form of attention. The context vector now interacts with the hidden state to produce attention weights, which are then used to calculate a weighted context representation from the encoder's outputs (`encoder_outputs`). The rest of the process remains autoregressive; the previously predicted token is still fed back for the next prediction, however now incorporates a dynamically adjusted contextual information.

For further exploration of this area, I would recommend consulting resources focusing on sequence-to-sequence learning, attention mechanisms, and recurrent neural network architectures. Look for materials outlining the theoretical foundations behind sequence modeling, specifically in natural language processing tasks. Texts detailing practical implementations of machine translation systems, especially those focusing on attention and transformer networks, would also be beneficial. Furthermore, research articles that discuss specific applications of these techniques, along with comparisons of different model types, would provide a deeper understanding of how recursion within the decoder plays a crucial role in the task of sequence generation. Specifically, resources from academia and the open-source deep learning community will prove highly informative.
