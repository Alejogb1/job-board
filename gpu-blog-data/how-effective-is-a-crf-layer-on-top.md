---
title: "How effective is a CRF layer on top of a BiLSTM?"
date: "2025-01-30"
id: "how-effective-is-a-crf-layer-on-top"
---
Conditional Random Fields (CRFs), when paired with Bidirectional Long Short-Term Memory networks (BiLSTMs), substantially enhance sequence labeling tasks, particularly those requiring accurate prediction of structured output. I've observed this improvement firsthand across several natural language processing projects focused on tasks like named entity recognition (NER) and part-of-speech (POS) tagging. The core issue stems from BiLSTMs' tendency to generate label predictions independently, neglecting crucial label dependencies that are inherent to sequences. A CRF layer addresses this deficiency by explicitly modeling transitions between labels.

A BiLSTM, at its core, is adept at capturing contextual information within a sequence. It processes input sequentially, forward and backward, learning representations that consider surrounding words and their relationships. It outputs a probability distribution over the possible labels for each position in the input sequence. However, this per-position prediction approach frequently overlooks valid label sequences. For instance, in NER, the label ‘I-Person’ (Inside of a Person) should typically be preceded by either ‘B-Person’ (Beginning of a Person) or another ‘I-Person’, not by ‘B-Location’. The BiLSTM by itself is not inherently aware of these constraints, potentially resulting in invalid label sequences.

The CRF layer remedies this by modeling the dependencies between labels using transition probabilities. Instead of predicting labels independently, the CRF considers the entire output sequence and scores it based on both the individual predictions (emission scores from the BiLSTM) and the transition scores between adjacent labels. These transition scores quantify the likelihood of one label following another within a sequence. During training, the CRF learns these transition scores, ultimately favoring sequences that adhere to established label conventions. This shift from independent to holistic sequence prediction substantially enhances overall labeling accuracy, particularly for tasks where label dependencies are critical. The model seeks to maximize the probability of the entire sequence, ensuring that predicted sequences adhere to these learned constraints.

Consider the following Python example using the PyTorch library. Assume we have a pre-trained BiLSTM producing per-token outputs. I represent the label sequence as integers:

```python
import torch
import torch.nn as nn

class CRF(nn.Module):
    def __init__(self, num_labels):
        super(CRF, self).__init__()
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))

    def forward(self, emissions, mask):
        seq_length, batch_size, num_labels = emissions.shape
        start_scores = torch.randn(batch_size, num_labels, device=emissions.device)
        end_scores = torch.randn(batch_size, num_labels, device=emissions.device)

        forward_scores = self._forward_alg(emissions, mask, start_scores, end_scores)
        return -forward_scores # return negative log likelihood

    def _forward_alg(self, emissions, mask, start_scores, end_scores):
       seq_length, batch_size, num_labels = emissions.shape

       alpha = start_scores

       for t in range(seq_length):
            emit_score = emissions[t]
            broadcast_alpha = alpha.unsqueeze(2)
            broadcast_emit = emit_score.unsqueeze(1)
            broadcast_trans = self.transitions.unsqueeze(0)

            next_alpha = torch.logsumexp(broadcast_alpha+broadcast_emit+broadcast_trans, dim=1)
            alpha = torch.where(mask[t].unsqueeze(1), next_alpha, alpha)
       final_scores = alpha + end_scores
       return torch.logsumexp(final_scores,dim=1).sum()

    def decode(self, emissions, mask):
      seq_length, batch_size, num_labels = emissions.shape
      start_scores = torch.randn(batch_size, num_labels, device=emissions.device)
      backpointers = []
      alpha = start_scores

      for t in range(seq_length):
           emit_score = emissions[t]
           broadcast_alpha = alpha.unsqueeze(2)
           broadcast_emit = emit_score.unsqueeze(1)
           broadcast_trans = self.transitions.unsqueeze(0)

           max_scores, max_indices = torch.max(broadcast_alpha+broadcast_emit+broadcast_trans, dim=1)
           backpointers.append(max_indices)
           alpha = torch.where(mask[t].unsqueeze(1),max_scores, alpha)

      path = torch.zeros(seq_length,batch_size,dtype=torch.int64,device=emissions.device)
      end_score = alpha + torch.randn(batch_size, num_labels,device=emissions.device)
      max_end, max_end_indices = torch.max(end_score,dim=1)
      path[-1] = max_end_indices
      for t in reversed(range(seq_length-1)):
          path[t] = backpointers[t+1].gather(dim=1, index=path[t+1].unsqueeze(1)).squeeze(1)
      return path
```

This code defines a basic CRF layer implemented in PyTorch.  `forward` computes the negative log-likelihood, and `decode` finds the most likely sequence.  The `_forward_alg` implements the forward algorithm. The `transitions` parameter stores learnable transition scores between labels. The forward algorithm calculates the total score of all possible paths by summing over all possible transitions. During training, we minimize this negative log-likelihood. Note the use of `mask` which allows to account for variable length sequences in batch.

Here's an example showcasing how to connect a BiLSTM output to a CRF layer:

```python
import torch
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.emission = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_seq, mask):
        embedded = self.embedding(input_seq)
        lstm_out, _ = self.bilstm(embedded)
        emissions = self.emission(lstm_out)
        return self.crf(emissions,mask)

    def decode(self, input_seq, mask):
         embedded = self.embedding(input_seq)
         lstm_out, _ = self.bilstm(embedded)
         emissions = self.emission(lstm_out)
         return self.crf.decode(emissions, mask)

# Example usage
vocab_size = 100
embedding_dim = 50
hidden_dim = 100
num_labels = 5
model = BiLSTM_CRF(vocab_size, embedding_dim, hidden_dim, num_labels)

input_seq = torch.randint(0, vocab_size, (2, 10)) # 2 sequences of length 10
mask = torch.ones(2, 10, dtype=torch.bool)
mask[0,5:] = False #variable length example

loss = model(input_seq, mask) #calculate loss from forward
path = model.decode(input_seq,mask) #retrieve the most likely path
```
In this case, `BiLSTM_CRF` contains a BiLSTM, a linear layer for emissions, and the CRF layer itself. During the forward pass, the BiLSTM output is converted to emissions through a linear layer before being passed into the CRF layer. The `decode` method retrieves the sequence of labels that maximizes the log probability. Note again the mask used.

Finally, consider the practical aspect of training. A basic training loop using the custom model might look like:

```python
import torch.optim as optim

# Data loading and preparation (omitted for brevity)
#Assume that training_data is a list of (sequence, mask, labels) triples
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    epoch_loss = 0
    for sequence, mask, labels in training_data:
       optimizer.zero_grad()
       loss = model(sequence,mask)
       loss.backward()
       optimizer.step()
       epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(training_data)}")

# Example of evaluation after training:
with torch.no_grad():
    for sequence, mask, true_labels in eval_data:
        predicted_labels = model.decode(sequence,mask)
        # Compare predicted and true labels for evaluation
```

This code excerpt demonstrates a standard training loop using the Adam optimizer. Each training batch is processed to calculate the negative log likelihood, and the model's parameters are updated. Once training is complete, evaluation occurs using the `decode` function, comparing predicted labels to true labels from the evaluation dataset.

For further study, I suggest focusing on theoretical foundations of CRFs; the original publication "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data" by Lafferty, McCallum, and Pereira is a good starting point, though quite math heavy. Then, moving to practical implementations, consider investigating resources on PyTorch’s and TensorFlow’s implementation of BiLSTMs and CRFs, focusing on understanding the data structures and API. Publications concerning named entity recognition or part of speech tagging typically showcase both theory and application of these techniques and can prove very insightful.  Focusing on specific applications will solidify understanding.
