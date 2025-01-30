---
title: "How can target decoder inputs be handled for a self-attention transformer model's prediction phase?"
date: "2025-01-30"
id: "how-can-target-decoder-inputs-be-handled-for"
---
The autoregressive nature of Transformer decoders requires specific handling of target inputs during the prediction phase. Unlike the training phase where the entire target sequence is available, prediction necessitates feeding the decoder one token at a time, using the decoder's previous output as the next input. This sequential dependency directly impacts the input structure and flow within the decoder.

During training, the decoder receives a "shifted right" version of the target sequence. This means the target sequence `[y_1, y_2, y_3, ..., y_n]` is transformed into `[<start>, y_1, y_2, ..., y_{n-1}]`, and the model is tasked with predicting the next token `y_i` given all previous tokens. Masking ensures that the decoder cannot "peek" ahead at future tokens within this sequence. Specifically, a causal mask prevents information from tokens `y_j (j >= i)` from influencing the prediction of token `y_i`. However, in the prediction or inference phase, we do not have the full target sequence readily available. Instead, we must generate the target sequence incrementally.

This process begins with a start token, often denoted as `<start>`. This initial token is fed to the decoder. The decoder produces a probability distribution over the entire vocabulary for the first token in the target sequence. The token with the highest probability (or using other sampling methods) becomes the first predicted token, `\hat{y}_1`. This newly predicted token `\hat{y}_1` then becomes the *next* input to the decoder, appended after the `<start>` token. The decoder then generates a probability distribution for the *second* target token, and the process repeats until an end token `<end>` is generated, or a maximum sequence length is reached.

This incremental prediction loop is essential for effectively using a trained Transformer decoder in a practical sequence generation setting. The decoder’s self-attention mechanism now operates on an ever-growing sequence of predicted tokens, influencing subsequent predictions. Effectively, the decoder learns to generate outputs based on its own previously generated output, which was influenced by previously generated output, and so on.

Consider a language translation task, translating from English to French. Let’s say the English input is "The cat sat". The trained model, having an encoder and a decoder, processes the encoded English input. In prediction, the decoder receives `<start>` as its input and generates a distribution over the French vocabulary. Suppose the highest probability is associated with "Le". Now, the decoder is fed `<start>`, "Le". This time the decoder generates a distribution. Let’s say "chat" becomes the next most probable output. This process continues, the next input sequence being `<start>`, "Le", "chat", and so forth until the end of the French sentence is predicted (represented by an `<end>` token).

The absence of the 'shifted right' target sequence during prediction requires a dynamic handling of the decoder input and output during this process. Let’s examine a conceptual code implementation using a simplified example.

**Code Example 1: Basic Decoding Loop**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyDecoder(nn.Module): # Simplified for illustration purposes
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size) # Output layer
    
    def forward(self, input_seq):
      embedded = self.embedding(input_seq)
      out = self.linear(embedded)
      return out

def predict_sequence(model, start_token, max_length, device):
    input_sequence = torch.tensor([start_token]).unsqueeze(0).to(device) # [1, 1]
    output_sequence = []
    
    for _ in range(max_length):
        logits = model(input_sequence) # [1, seq_len, vocab_size] 
        next_token_probs = F.softmax(logits[:, -1, :], dim=1) # [1, vocab_size] get the last prediction only
        next_token = torch.argmax(next_token_probs, dim=1).item() # Index of the most probable token
        
        output_sequence.append(next_token)
        input_sequence = torch.cat((input_sequence, torch.tensor([[next_token]]).to(device)), dim=1) # [1, seq_len+1]
        
        if next_token == end_token:
            break
    return output_sequence

# Example Usage (placeholders, vocab size=100, embedding_dim=32, hidden_dim=64)
vocab_size = 100
embedding_dim = 32
hidden_dim = 64
start_token = 1 # Example start token index
end_token = 2 # Example end token index
max_length = 50
device = torch.device("cpu") # Or "cuda"

model = DummyDecoder(vocab_size, embedding_dim, hidden_dim).to(device)

predicted_sequence = predict_sequence(model, start_token, max_length, device)
print(f"Predicted sequence: {predicted_sequence}")

```

This first example outlines the core functionality of the iterative prediction process. Note that the `DummyDecoder` is a simplification and does not contain actual transformer layers for conciseness. The `predict_sequence` function initiates with a single start token and proceeds to predict subsequent tokens, adding the newly predicted token to the input sequence. The loop terminates either after reaching the max sequence length, or encountering the end token.

**Code Example 2: Adding a Mask and Temperature for Prediction**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedDummyDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq, mask):
      embedded = self.embedding(input_seq)
      out = self.linear(embedded)
      return out
    
def create_causal_mask(size, device):
    mask = torch.ones(size, size, dtype=torch.bool).to(device).triu(diagonal=1) # 1 if cannot see, 0 otherwise
    return mask

def predict_sequence_masked(model, start_token, max_length, device, temperature=1.0):
    input_sequence = torch.tensor([start_token]).unsqueeze(0).to(device)
    output_sequence = []
    
    for _ in range(max_length):
      seq_len = input_sequence.size(1)
      mask = create_causal_mask(seq_len, device)
      logits = model(input_sequence, mask)
      
      next_token_probs = F.softmax(logits[:, -1, :] / temperature, dim=1)
      next_token = torch.multinomial(next_token_probs, num_samples=1).item()
      
      output_sequence.append(next_token)
      input_sequence = torch.cat((input_sequence, torch.tensor([[next_token]]).to(device)), dim=1)
      
      if next_token == end_token:
          break
    return output_sequence

# Example Usage
vocab_size = 100
embedding_dim = 32
hidden_dim = 64
start_token = 1
end_token = 2
max_length = 50
device = torch.device("cpu")

model = MaskedDummyDecoder(vocab_size, embedding_dim, hidden_dim).to(device)

predicted_sequence = predict_sequence_masked(model, start_token, max_length, device, temperature=0.8) # added temperature
print(f"Predicted sequence with mask and temp: {predicted_sequence}")
```

Here, we introduce a causal mask to demonstrate that it can also be applied during decoding. Even though it's not needed since there are not future tokens to mask, it's a demonstration of how masking works, and also how the decoder architecture can be generalized to use it at prediction time too. This version also implements temperature sampling which is used to control the diversity of the output. Higher temperature values lead to more randomness, while lower values produce more deterministic outputs. It replaces the argmax with multinomial sampling, incorporating randomness based on the temperature parameter.

**Code Example 3: Using Beam Search**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeamSearchDummyDecoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

  def forward(self, input_seq):
      embedded = self.embedding(input_seq)
      out = self.linear(embedded)
      return out

def beam_search(model, start_token, max_length, device, beam_width=3):
    sequences = [([start_token], 0.0)] # (sequence, log probability)
    
    for _ in range(max_length):
        all_candidates = []
        for seq, log_prob in sequences:
            input_seq = torch.tensor(seq).unsqueeze(0).to(device)
            logits = model(input_seq)
            next_token_probs = F.log_softmax(logits[:, -1, :], dim=1)

            top_k_values, top_k_indices = torch.topk(next_token_probs, beam_width, dim=1)
            
            for i in range(beam_width):
                next_token = top_k_indices[0, i].item()
                new_log_prob = log_prob + top_k_values[0, i].item()
                all_candidates.append((seq + [next_token], new_log_prob))
        
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

        if any(seq[-1] == end_token for seq, _ in sequences):
           return [seq for seq, _ in sequences if seq[-1] == end_token][0] # return first that hits end token

    return sequences[0][0] # return best after loop is done

# Example Usage
vocab_size = 100
embedding_dim = 32
hidden_dim = 64
start_token = 1
end_token = 2
max_length = 50
device = torch.device("cpu")

model = BeamSearchDummyDecoder(vocab_size, embedding_dim, hidden_dim).to(device)

predicted_sequence = beam_search(model, start_token, max_length, device, beam_width=5) # added beam width
print(f"Predicted sequence with beam search: {predicted_sequence}")

```
The third example introduces beam search, a technique used to enhance the quality of the generated sequence. Unlike greedy decoding, which selects the single most probable next token, beam search keeps track of the top-`k` candidate sequences. It explores several potential token choices at each step. This allows the model to potentially generate sequences with higher overall likelihood. This example also modifies the decoder to avoid masking as it is unecessary. It uses log probabilities for numerical stability and provides an example of early stop when the beam finds the end token.

For further understanding, I recommend exploring material focusing on sequence-to-sequence models, attention mechanisms, and autoregressive generation. Research papers on the Transformer architecture itself will also provide essential insights. Additionally, various tutorials covering PyTorch's neural network implementation are invaluable. Consider consulting material on inference methods in sequence models, such as beam search and sampling.
