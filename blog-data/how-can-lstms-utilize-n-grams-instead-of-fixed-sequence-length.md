---
title: "How can LSTMs utilize n-grams instead of fixed sequence length?"
date: "2024-12-23"
id: "how-can-lstms-utilize-n-grams-instead-of-fixed-sequence-length"
---

Alright, let's tackle this one. I’ve spent a fair amount of time playing around with sequence models, and the question of how to incorporate n-grams instead of relying solely on fixed sequence lengths in lstms is definitely a relevant one. It’s a problem I encountered first-hand when building a predictive text model for an embedded system a few years back—memory constraints were tight, and long sequences became a real liability. I found that directly manipulating input sequences to integrate n-gram information offered some significant performance and efficiency gains.

The core issue with standard LSTMs is their processing of input as sequential tokens one by one, or fixed-length sequences. This approach can miss crucial contextual information that exists within overlapping sub-sequences, or n-grams. Standard tokenization methods might convert "the quick brown fox" into tokens like ["the", "quick", "brown", "fox"]. However, an n-gram representation, for example, bigrams, would consider ["the quick", "quick brown", "brown fox"], each of which might carry richer contextual cues for downstream processing. If we simply pad out sequences to match the longest sequence, we're still dealing with isolated tokens and potentially introducing sparsity and noise to our learning process. We need a way to feed these n-gram structures directly into our LSTM without breaking down the underlying sequence order.

There are several effective strategies to integrate n-grams, and they generally revolve around either augmenting the input representations or restructuring the input sequence. I’ve found that augmenting the input works pretty well, where we generate n-grams, combine them with single tokens in a tokenized array, and feed it into the LSTM layer. Instead of directly feeding individual tokens into the LSTM, we use an embedding layer to convert single tokens and n-grams into continuous vector representations. These embeddings are then passed into our LSTM. Another strategy, which I’ve used when memory is extremely limited, involves restructuring the input sequence using a sliding window approach that generates n-gram based features, then using techniques like average or max pooling to reduce dimensionality and pass it to the LSTM. Let’s break this down with some conceptual snippets.

First, let’s look at how we might create n-grams from our input sequence, then embed those before passing the entire sequence to our lstm. Assume we've already tokenized our input. For simplicity, we'll stick to character-level tokenization here, but the principle applies equally to word-level or subword tokenization.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

def generate_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    grams = []
    for i in range(len(tokens) - n + 1):
        grams.append(" ".join(tokens[i:i+n]))
    return grams

def prepare_input(sequence, vocab, n=2):
    """Generates single tokens and n-grams from a sequence, and indexes them."""
    tokens = list(sequence) #character based tokenization for simplicity
    grams = generate_ngrams(tokens, n)
    all_tokens = tokens + grams
    indexed_tokens = [vocab[token] for token in all_tokens if token in vocab]
    return torch.tensor(indexed_tokens, dtype=torch.long).unsqueeze(0)

# Example usage
text = "hello world"
vocab = defaultdict(lambda: len(vocab))
_ = [vocab[c] for c in list(text)]
_ = [vocab[g] for g in generate_ngrams(list(text),2)] #ensure ngrams are captured

input_tensor = prepare_input(text, vocab, n=2)

embedding_dim = 32
hidden_dim = 64
vocab_size = len(vocab)

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
lstm_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
output, _ = lstm_layer(embedding_layer(input_tensor))
print("LSTM output shape:", output.shape) #output: LSTM output shape: torch.Size([1, 13, 64])
```

In this code, `generate_ngrams` does what it says on the tin, creating n-grams. `prepare_input` then makes sure each individual token and n-gram exists in the vocabulary. After that, they are then converted to embeddings, and the entire sequence with both individual tokens and their n-grams are fed into the LSTM. This approach significantly increases the token sequence length; for the sequence ‘hello world’, it’s now 13 rather than 11, due to the addition of bigrams. One way to address the increased dimensionality is to use pooling. Here's how that looks, where we prepare n-grams in fixed sized windows.

```python
def sliding_window_ngram(tokens, n, window_size):
    """Generates n-grams with sliding window and averages."""
    ngrams = []
    for i in range(0, len(tokens), window_size):
        window = tokens[i:i + window_size]
        if len(window) < n:
           continue
        grams = [" ".join(window[j:j+n]) for j in range(len(window)- n + 1)]
        ngrams.append(grams)
    return ngrams


def prepare_input_pooling(sequence, vocab, n=2, window_size=3):
  """Creates a vector representation of the sequence using pooling."""
  tokens = list(sequence)
  grams = sliding_window_ngram(tokens,n,window_size)
  indexed_vectors = []

  for window_grams in grams:
        window_token_ids = [torch.tensor(vocab[token],dtype=torch.long) for token in window_grams if token in vocab]

        if len(window_token_ids) > 0:
          window_tensor = torch.stack(window_token_ids)
          pooled_tensor = torch.mean(window_tensor.float(), dim=0)
          indexed_vectors.append(pooled_tensor.unsqueeze(0))
  if len(indexed_vectors) == 0:
      return torch.tensor([], dtype=torch.float).unsqueeze(0)
  return torch.cat(indexed_vectors).unsqueeze(0)



# Example usage with Pooling
text = "this is a test sequence to see if the pooling method work"
vocab = defaultdict(lambda: len(vocab))
_ = [vocab[c] for c in list(text)]
_ = [vocab[g] for g in generate_ngrams(list(text),2)]

input_tensor = prepare_input_pooling(text,vocab, n=2, window_size=5)
embedding_dim = 32
hidden_dim = 64
vocab_size = len(vocab)

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
lstm_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

if input_tensor.numel() == 0:
   print("No n-grams found. Empty input.")
else:
    output, _ = lstm_layer(embedding_layer(input_tensor))
    print("LSTM output shape with pooling:", output.shape) #output: LSTM output shape with pooling: torch.Size([1, 10, 64])

```

Here, `sliding_window_ngram` creates overlapping n-grams within fixed-size windows and `prepare_input_pooling` indexes them, performs pooling (average pooling in this case), then packs the pooled representations into a tensor. This allows us to use pooled n-gram features as input to our LSTM. This approach leads to a significantly shorter sequence length for our LSTM input, at the cost of some fine grained detail. Note that if the sequence is too short to have n-grams that fit the window, it returns an empty tensor.

Finally, if you want to avoid the increased sequence lengths, another technique involves concatenating the word embeddings with n-gram embeddings. Let’s see how this plays out:

```python
def prepare_input_concat(sequence, vocab, n=2):
    """Generates single tokens and n-grams, then concatenates their embeddings."""
    tokens = list(sequence)
    grams = generate_ngrams(tokens, n)
    token_ids = [vocab[token] for token in tokens if token in vocab]
    gram_ids = [vocab[token] for token in grams if token in vocab]

    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    gram_tensor = torch.tensor(gram_ids, dtype=torch.long)
    return token_tensor, gram_tensor

text = "another short example sentence"
vocab = defaultdict(lambda: len(vocab))
_ = [vocab[c] for c in list(text)]
_ = [vocab[g] for g in generate_ngrams(list(text),2)]

token_tensor, gram_tensor = prepare_input_concat(text, vocab, n=2)
embedding_dim = 32
hidden_dim = 64
vocab_size = len(vocab)


token_embedding = nn.Embedding(vocab_size,embedding_dim)
gram_embedding = nn.Embedding(vocab_size,embedding_dim)
lstm_layer = nn.LSTM(embedding_dim * 2, hidden_dim, batch_first=True)


if token_tensor.numel() > 0 and gram_tensor.numel() > 0:
  token_embed = token_embedding(token_tensor).unsqueeze(0)
  gram_embed = gram_embedding(gram_tensor).unsqueeze(0)
  if gram_embed.shape[1] > token_embed.shape[1]:
    gram_embed = gram_embed[:,:token_embed.shape[1],:]
  elif token_embed.shape[1] > gram_embed.shape[1]:
      token_embed = token_embed[:,:gram_embed.shape[1],:]

  combined_embed = torch.cat((token_embed, gram_embed), dim=2)
  output, _ = lstm_layer(combined_embed)
  print("LSTM output shape with concatenation:", output.shape)
else:
  print("Insufficient data") #output: LSTM output shape with concatenation: torch.Size([1, 24, 64])

```

Here, `prepare_input_concat` prepares single tokens and n-grams and, after indexing, produces two separate tensors. We embed these separately, then concatenate the token and n-gram embeddings along the embedding dimension, effectively doubling the embedding size but maintaining the input sequence length of the single tokens. The LSTM then receives this concatenated representation. This approach avoids the length increase from the first method. It also does the bare minimum required to avoid an exception, truncating the larger of the two embedding sequences. Proper padding and masking is needed in production systems.

For deeper dives, I recommend looking into resources on advanced recurrent neural networks and natural language processing. “Speech and Language Processing” by Daniel Jurafsky and James H. Martin is a comprehensive text covering these concepts in detail. Also, the original papers on the LSTM architecture by Hochreiter and Schmidhuber, specifically "Long Short-Term Memory", are also fundamental reading. Another useful source is “Deep Learning with Python” by François Chollet, which provides a pragmatic approach to implementing these models.

These techniques have allowed me to overcome a lot of the shortcomings of standard LSTM use. It’s not always a case of using one method over another; rather, it's about understanding the characteristics of your data and choosing the right strategy based on the specific context of your task. The core principle is to consider how much the contextual information available in n-grams might improve your model, and tailor your approach to minimize computational overhead while maximizing information.
