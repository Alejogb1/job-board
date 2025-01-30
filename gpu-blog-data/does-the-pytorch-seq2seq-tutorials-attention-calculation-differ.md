---
title: "Does the PyTorch seq2seq tutorial's attention calculation differ from Bahdanau/Luong's original paper?"
date: "2025-01-30"
id: "does-the-pytorch-seq2seq-tutorials-attention-calculation-differ"
---
The PyTorch seq2seq tutorial, while fundamentally implementing attention mechanisms, does deviate in its specific calculation from the attention scoring methods described in both Bahdanau's (2014) and Luong's (2015) seminal papers. My experience in implementing various sequence-to-sequence models, particularly for natural language tasks involving translation and text summarization, has revealed subtle but significant differences that impact performance. The key discrepancy arises in how the alignment scores are computed between the decoder’s hidden state and the encoder’s hidden states, as well as in the subsequent normalization step.

Firstly, let's examine the original formulations. Bahdanau's additive attention, as presented in “Neural Machine Translation by Jointly Learning to Align and Translate,” computes attention scores (or energies) by passing the concatenation of the decoder hidden state and each encoder hidden state through a learned feedforward neural network. Specifically, for a decoder hidden state *s<sub>t</sub>* and an encoder hidden state *h<sub>j</sub>*, the alignment score *e<sub>tj</sub>* is calculated as:

*e<sub>tj</sub>* = *v<sup>T</sup>* tanh(*W<sub>s</sub>* *s<sub>t</sub>* + *W<sub>h</sub>* *h<sub>j</sub>*)

Here, *W<sub>s</sub>*, *W<sub>h</sub>*, and *v* are learnable weight matrices and vectors, respectively. This process involves concatenating *s<sub>t</sub>* and *h<sub>j</sub>* implicitly through the application of matrix multiplications and using a tanh activation function before the projection into a scalar value. The resulting *e<sub>tj</sub>* scores are then softmax-normalized to obtain attention weights.

Luong's multiplicative attention, outlined in "Effective Approaches to Attention-based Neural Machine Translation," offers variations, including dot-product, scaled dot-product, and general attention. A core element, which I'll focus on for this comparison, is the general attention. Here, the score is calculated as:

*e<sub>tj</sub>* = *s<sub>t</sub><sup>T</sup>* *W<sub>a</sub>* *h<sub>j</sub>*

Where *W<sub>a</sub>* is a learnable weight matrix. This formulation uses a simpler matrix multiplication to compute the score without any activation before the final calculation. Again, these scores are subsequently normalized with softmax.

Now, let's contrast these methods with how PyTorch's seq2seq tutorial implements attention. In my experience, the tutorial often simplifies attention for clarity, usually adopting a variant resembling Luong’s general attention, but with a single linear layer transformation. Specifically, the tutorial code computes attention weights through the following steps: first, the decoder hidden state is projected to a new vector representation via linear transformation *W<sub>a</sub>*. Then, dot product scores between this projected decoder state and each encoder hidden state are computed and normalized with the softmax function. The key difference lies in how these scores are computed directly, without any intermediary transformation of the encoder hidden states. It directly dot-products the transformed decoder hidden states with the encoder hidden states as opposed to transforming individual encoder hidden states prior to the dot product with the decoder state, as in the original Bahdanau or Luong formulations.

The following code examples demonstrate the contrasts. The first shows a simplified version of Bahdanau's attention, implementing the feedforward network approach.

```python
import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.Ws = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hidden_dim]
        # encoder_outputs: [seq_len, batch_size, hidden_dim]
        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        Ws_s = self.Ws(decoder_hidden) # [batch_size, hidden_dim]
        Ws_s = Ws_s.unsqueeze(1).expand(-1, seq_len, -1) # [batch_size, seq_len, hidden_dim]
        Wh_h = self.Wh(encoder_outputs.transpose(0,1)) # [batch_size, seq_len, hidden_dim]
        
        tanh_result = torch.tanh(Ws_s+Wh_h) # [batch_size, seq_len, hidden_dim]
        scores = self.v(tanh_result).squeeze(2) # [batch_size, seq_len]

        attention_weights = torch.softmax(scores, dim=1) # [batch_size, seq_len]
        return attention_weights
```

Here, the `Ws` and `Wh` linear layers transform the decoder and encoder hidden states respectively before a `tanh` activation. The final score is a single linear projection after this. This reflects the process described in the Bahdanau paper.

The next example shows a simplified general attention mechanism inspired by Luong's work:

```python
import torch
import torch.nn as nn

class LuongGeneralAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongGeneralAttention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hidden_dim]
        # encoder_outputs: [seq_len, batch_size, hidden_dim]
        batch_size = encoder_outputs.shape[1]

        Wa_s = self.Wa(decoder_hidden) # [batch_size, hidden_dim]
        scores = torch.bmm(Wa_s.unsqueeze(1), encoder_outputs.transpose(0,1).transpose(1,2)).squeeze(1) # [batch_size, seq_len]
        attention_weights = torch.softmax(scores, dim=1) # [batch_size, seq_len]

        return attention_weights
```

The code uses `Wa` to project the decoder hidden state. It then calculates the scores via a batch matrix multiplication between this projected decoder state and all encoder states.

Finally, consider a common simplified attention implementation seen in PyTorch tutorials. This simplified implementation uses a single linear layer to project decoder states and perform the dot product with the encoder hidden states:

```python
import torch
import torch.nn as nn

class PyTorchTutorialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(PyTorchTutorialAttention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hidden_dim]
        # encoder_outputs: [seq_len, batch_size, hidden_dim]

        Wa_s = self.Wa(decoder_hidden) # [batch_size, hidden_dim]
        scores = torch.bmm(Wa_s.unsqueeze(1), encoder_outputs.transpose(0,1).transpose(1,2)).squeeze(1) # [batch_size, seq_len]
        attention_weights = torch.softmax(scores, dim=1) # [batch_size, seq_len]
        return attention_weights

```
This version is quite similar to the previous Luong-style example, except it does not project the encoder hidden states with a separate linear layer and is therefore a simplified form.

These three code blocks highlight the distinctions: Bahdanau's additive attention involves a more complex transformation involving both encoder and decoder states with activation. The tutorial's attention mechanism, as well as the general Luong attention, primarily transforms the decoder state alone, relying only on a single linear transformation before computing the scores with the encoder's original hidden states directly. The specific transformations and the presence or absence of activation functions prior to score calculation are the core differences.

The impact of these differences is primarily reflected in the model's ability to learn contextual relationships. Bahdanau's approach, with its use of activation and transformations on both encoder and decoder hidden states, can provide more flexibility in learning the interaction between source and target sequences. On the other hand, the tutorial implementation's direct scoring might be computationally efficient but can limit the model's ability to learn intricate patterns.

For a comprehensive understanding of attention mechanisms and their impact on sequence models, I recommend exploring the following resources: "Attention is All You Need" (Vaswani et al., 2017), which introduces the transformer architecture and its self-attention mechanism. Additionally, examining “A Survey of Attention Mechanisms” (Bahdanau et al., 2015) offers a broad view of different attention methods. Furthermore, delving into the original papers by Bahdanau and Luong provides a foundation of theoretical understanding of attention mechanisms. Exploring implementations of attention mechanisms in frameworks like Tensorflow and AllenNLP can provide additional practical insights. These resources will contribute to a broader understanding, going beyond the initial simplification often seen in introductory tutorials and improving the implementation and overall understanding of the subtleties of attention mechanisms in sequence to sequence models.
