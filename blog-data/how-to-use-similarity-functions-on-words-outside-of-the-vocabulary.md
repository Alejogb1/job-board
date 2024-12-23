---
title: "How to use similarity functions on words outside of the vocabulary?"
date: "2024-12-16"
id: "how-to-use-similarity-functions-on-words-outside-of-the-vocabulary"
---

, let's tackle this one. It's a situation I’ve definitely encountered more than a few times, especially back when I was working on that large-scale text analysis project for the legal firm, where we had a particularly messy dataset riddled with typos and neologisms. Dealing with out-of-vocabulary (OOV) words when using similarity functions is, shall we say, a non-trivial task, but it’s absolutely manageable with a structured approach.

Essentially, the problem is this: similarity functions, be it cosine similarity on word vectors or edit distance, are designed to work with known vocabulary. When a word pops up that’s not in our model's dictionary, those standard calculations simply fall flat. We're essentially asking the model to understand something it hasn't been trained to understand directly. This leads to either completely missing the similarities or generating incorrect and misleading results. But fear not, there are ways around this, and some are more effective than others depending on the context.

First, let’s consider the options. A naive approach would be to simply drop these OOV words, but this can lead to a massive loss of information. In many cases, the OOV word itself is critical to the meaning of the sentence. The first technique I often employ involves utilizing character-level encodings. Instead of treating a word as a single unit, we break it down into smaller, more fundamental parts like characters or sub-word units (e.g., byte-pair encoding or word-pieces). This allows us to create representations for OOV words by combining representations of these sub-word components. Think of it as building a word from a set of Lego blocks rather than needing a single, pre-made structure.

Here’s an example of generating a character-based average embedding for an OOV word using python and numpy:

```python
import numpy as np

def character_level_embedding(word, char_embeddings, unknown_embedding):
    """Generates an average character embedding for a word."""
    if not word:
      return unknown_embedding #handle edge case for empty strings

    word_embedding = np.zeros(char_embeddings['a'].shape) #assuming all embeddings have same dimension
    valid_char_count = 0
    for char in word:
        if char in char_embeddings:
            word_embedding += char_embeddings[char]
            valid_char_count += 1
        else:
            word_embedding += unknown_embedding

    if valid_char_count == 0:
      return unknown_embedding
    return word_embedding / len(word) #average embedding

# Example usage - let's pretend we have some char embeddings
char_embeddings = {
    'a': np.array([0.1, 0.2, 0.3]),
    'b': np.array([0.4, 0.5, 0.6]),
    'c': np.array([0.7, 0.8, 0.9]),
    'd': np.array([0.2, 0.1, 0.4]),
    'e': np.array([0.3, 0.6, 0.7])
}

unknown_embedding = np.array([0.0, 0.0, 0.0])

oov_word = 'abcde' # in-vocabulary
oov_word_2 = 'abzcd' # out-of-vocabulary with 'z' unknown
oov_word_3 = 'xyz' # out of vocabulary
print(character_level_embedding(oov_word, char_embeddings, unknown_embedding))
print(character_level_embedding(oov_word_2, char_embeddings, unknown_embedding))
print(character_level_embedding(oov_word_3, char_embeddings, unknown_embedding))

```
In this snippet, I illustrate how to create an embedding for both in-vocabulary and out-of-vocabulary words using simple averaging. The important part is that even if a character is not found, we can still generate a representation based on the parts we know and, if needed, by incorporating a predefined unknown or zero-filled embedding. In practice, you'd use real pre-trained character embeddings.

Another robust technique is to employ sub-word tokenization algorithms, like byte-pair encoding (BPE) or word pieces. These methods learn common sub-word units from a corpus, allowing you to break down a rare or OOV word into smaller, known sub-word units. For example, “unbelievable” might be broken into "un", "believe", and "able". If these sub-words are present in our vocabulary, then the final embedding can be derived. This is especially useful when dealing with morphological variations of words. I vividly recall using this approach when tackling misspellings on product names when I worked for the e-commerce platform and the effectiveness was significant.

Here’s a Python code snippet using huggingface’s transformers library to illustrate BPE tokenization:

```python
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def bpe_embedding(word, tokenizer, embedding_matrix, unknown_embedding):
    """Generates an average BPE embedding for a word."""
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    if not token_ids:
      return unknown_embedding #handle empty token ids

    word_embedding = np.zeros(embedding_matrix.shape[1]) # match vector size of embedding
    valid_token_count = 0

    for token_id in token_ids:
      if token_id < len(embedding_matrix):
          word_embedding += embedding_matrix[token_id]
          valid_token_count += 1
      else:
        word_embedding += unknown_embedding

    if valid_token_count == 0:
       return unknown_embedding
    return word_embedding / len(tokens)

# Simulating a small embedding matrix
embedding_size = 768 # typical bert embedding size
vocab_size = tokenizer.vocab_size
embedding_matrix = np.random.rand(vocab_size, embedding_size)

unknown_embedding = np.zeros(embedding_size)


oov_word = 'unbelievable' # out-of-vocabulary to base bert
in_vocab_word = "apple"

print(bpe_embedding(oov_word, tokenizer, embedding_matrix, unknown_embedding))
print(bpe_embedding(in_vocab_word, tokenizer, embedding_matrix, unknown_embedding))

```

This example showcases how to use BPE tokenization, and demonstrates how, even with bert, the embeddings are combined. The key is we’re now operating on the sub-word level which, while not perfect, can result in far better outcomes.

Finally, there are more sophisticated methods that combine both character-level information and sub-word units, typically through neural networks. For example, you could train a model to predict the embedding of a word from its character sequences or sub-word tokens using, say, a recurrent network or a transformer model specifically designed for this. You can then use this model to generate embeddings for OOV words. This method is more computationally expensive, but typically yields superior results, especially for complex languages or domains. I used this approach to improve text categorization when working on a project about medical jargon, which constantly had new and uncommon terms. Here is a simplified example of how it works with a simple feed-forward network and embedding:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class WordEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WordEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # Average embeddings
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_oov_embedding(word, char_to_idx, encoder_model, embedding_dim):
    chars_idx = [char_to_idx.get(char, char_to_idx['<unk>']) for char in word]
    if not chars_idx:
        return torch.zeros(embedding_dim) # handle edge case

    input_tensor = torch.tensor(chars_idx).unsqueeze(0)
    with torch.no_grad():
        oov_embedding = encoder_model(input_tensor)

    return oov_embedding.squeeze()

# Generate a dictionary of characters to index and model setup
chars = ['<pad>', '<unk>'] + list("abcdefghijklmnoprstuvwxyz") # adding pad and unknown tokens
char_to_idx = {char: idx for idx, char in enumerate(chars)}
vocab_size = len(chars)
embedding_dim = 64
hidden_dim = 128
model = WordEncoder(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
# Simulate training data and training loop
for i in range (10): #simple training with dummy data to exemplify the encoder model
  input_chars = list("abc")
  target_chars = list("acb")
  input_indices = [char_to_idx[char] for char in input_chars ]
  target_indices = [char_to_idx[char] for char in target_chars ]
  input_tensor = torch.tensor(input_indices).unsqueeze(0)
  target_tensor = torch.tensor(target_indices).unsqueeze(0)
  optimizer.zero_grad()
  oov_embedding_model = model(input_tensor)
  target_embeddings = torch.mean(model.embedding(target_tensor), dim=1) #get desired target as an average of the embedded chars
  loss = criterion(oov_embedding_model, target_embeddings)
  loss.backward()
  optimizer.step()

oov_word = 'xyz' #out-of-vocabulary example
in_vocab_word = "abc"
print(generate_oov_embedding(oov_word, char_to_idx, model,embedding_dim ))
print(generate_oov_embedding(in_vocab_word, char_to_idx, model, embedding_dim))
```
This example, while simplified, demonstrates how a model could be trained to derive embeddings of words, including OOV words, based on character embeddings. I must stress, this method requires good quality character or sub-word embeddings that are tuned to the dataset in question.

For a deeper understanding of this topic, I’d highly recommend the following resources: "Speech and Language Processing" by Daniel Jurafsky and James H. Martin – it provides a solid theoretical grounding. The "Deep Learning" book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville is another key text. For more practical and modern approaches, the documentation for libraries like TensorFlow and PyTorch is very helpful. Pay particular attention to the modules related to embeddings, tokenization, and neural network layers, as well as the papers that introduced Byte Pair Encoding (BPE) and Word Piece algorithms – these are fundamental for subword models.

In summary, dealing with OOV words when using similarity functions is a complex problem. However, leveraging character-level encodings, sub-word tokenization, or training specialized models provides pragmatic and effective means to address this common issue. As always, the specific approach you choose will depend on your available resources, the characteristics of the dataset and the computational constraints, but the key is always to make sure that your method handles not only known words but also the unknown.
