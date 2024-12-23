---
title: "How can Indonesian tokenizers handle compound words?"
date: "2024-12-23"
id: "how-can-indonesian-tokenizers-handle-compound-words"
---

Alright, let's talk about Indonesian tokenization and those tricky compound words. I remember back in my early days working on a multilingual sentiment analysis project, the Indonesian component nearly brought our system to its knees precisely because of this issue. It’s not just about splitting text into words; it's about understanding the nuances of the language.

The problem with Indonesian, as with many agglutinative languages, lies in the formation of compound words. Indonesian frequently combines morphemes to create new words with often complex meanings. These combinations aren't always straightforward concatenations either; they can involve infixes, suffixes, and reduplication, making simple whitespace tokenization a disaster. Think of something like "mempertanggungjawabkan"— try splitting *that* naively and you'll end up with a semantic mess.

The approach to tackling this involves a combination of linguistic rules and statistical models, and realistically, no single method is a panacea. First, we can’t ignore the importance of manually crafted rules. This involves creating a comprehensive dictionary of common prefixes, suffixes, and infixes, plus a list of known compound words that are typically treated as a single token. This is where domain knowledge is critical; for example, terms used in legal text might be different than those used in social media.

Here's how a basic rule-based system could function conceptually, and we'll use a Python-like pseudocode to illustrate:

```python
def rule_based_tokenize(text, lexicon):
    tokens = []
    current_word = ""
    for char in text:
        if char.isspace():
             if current_word:
                 tokens.append(current_word)
                 current_word = ""
        else:
           current_word += char

    if current_word:
         tokens.append(current_word)


    processed_tokens = []
    for token in tokens:
      found_compound = False
      for compound_word in lexicon['compound_words']:
        if token == compound_word:
          processed_tokens.append(token)
          found_compound = True
          break

      if found_compound:
        continue

      for prefix in lexicon['prefixes']:
        if token.startswith(prefix):
            remaining = token[len(prefix):]
            if remaining in lexicon['base_words']:
                processed_tokens.append(prefix)
                processed_tokens.append(remaining)
                found_compound = True
                break

      if found_compound:
         continue

      for suffix in lexicon['suffixes']:
          if token.endswith(suffix):
             remaining = token[:-len(suffix)]
             if remaining in lexicon['base_words']:
                 processed_tokens.append(remaining)
                 processed_tokens.append(suffix)
                 found_compound = True
                 break

      if not found_compound:
            processed_tokens.append(token)
    return processed_tokens


lexicon = {
 'compound_words': ['beritahukan', 'bertanggungjawab'],
 'prefixes': ['mem','di','ter','ber','ke','se'],
 'suffixes' : ['kan', 'i'],
 'base_words': ['beri','tanggung','jawab']
}
text = "Saya harus mempertanggungjawabkan semua ini. tolong beritahukan dia"
tokens = rule_based_tokenize(text, lexicon)
print(tokens)
#expected output: ['Saya', 'harus', 'mem', 'pertanggung', 'jawab', 'kan', 'semua', 'ini.', 'tolong', 'beritahukan', 'dia']

```

This pseudocode is illustrative. We have a `rule_based_tokenize` function that uses a manually prepared lexicon with lists of compound words, common prefixes, suffixes, and a list of base words. It first performs basic whitespace tokenization, and then attempts to match or split into compound tokens based on the lexicon rules. Note that in real-world usage, you'd typically build a much larger and more nuanced lexicon, potentially with morphological analyzers too. It also wouldn't be this basic, and would deal with more complexity.

The challenge with this approach, however, is scalability and coverage. Manual rules can be time-consuming to create and might not cover all the variations, especially when new words and word combinations constantly emerge. This is where Statistical models come in, particularly those based on subword tokenization like Byte-Pair Encoding (BPE). BPE learns a subword vocabulary by iteratively merging frequently occurring character sequences in the training data. This allows the model to effectively handle out-of-vocabulary (OOV) words and compound words without needing to explicitly define all the rules.

Here's a simplified BPE concept illustrated in Python:

```python
import re
from collections import defaultdict

def get_pairs(tokens):
    pairs = defaultdict(int)
    for token in tokens:
        for i in range(len(token) - 1):
            pairs[token[i:i+2]] += 1
    return pairs

def merge_pairs(pairs, vocabulary):
   best_pair = max(pairs, key=pairs.get)
   new_vocabulary = []
   for token in vocabulary:
      new_token = re.sub(f'{re.escape(best_pair[0])}{re.escape(best_pair[1])}', f'{best_pair[0]}{best_pair[1]}', token)
      new_vocabulary.append(new_token)
   return new_vocabulary, best_pair


def bpe_tokenize(text, num_merges=100):
    text = text.lower()
    tokens = text.split()
    vocabulary = [' '.join(list(token)) for token in tokens]

    for _ in range(num_merges):
      pairs = get_pairs(vocabulary)
      if not pairs:
         break
      vocabulary, best_pair = merge_pairs(pairs, vocabulary)


    final_tokens = []
    for token in vocabulary:
        final_tokens.extend(token.split())
    return final_tokens

text = "mempertanggungjawabkan adalah kewajiban setiap warga negara.  "
tokens = bpe_tokenize(text, 20)
print(tokens)
# Expected Output (will vary due to random initialization): ['m', 'em', 'per', 'tanggung', 'ja', 'wab', 'k', 'an', 'adalah', 'ke', 'wa', 'jib', 'an', 'se', 'ti', 'ap', 'war', 'ga', 'ne', 'ga', 'ra.']

```

In this illustration, `bpe_tokenize` performs the BPE algorithm for a simplified text corpus. It starts by splitting the text into initial tokens, and then, iteratively merges the most frequently occurring character pairs into a new vocabulary set. After a predefined number of merges, the final vocabulary can be used for tokenization. Although it will not capture all complex words correctly and the output will be sensitive to `num_merges`, it gives a very rough idea of how BPE works. In practice, you'd train this on a large Indonesian corpus and would use more efficient implementations for production.

Finally, there's another important technique, especially in the context of deep learning, and this is the use of character-level or subword-level models, where individual characters, or groups of characters are used as tokens. This inherently handles compound words, since model learns representations of frequent combinations without pre-defined knowledge about Indonesian morphology.

Here's how a conceptual implementation might look like:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CharLevelLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(CharLevelLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(output[-1, :, :])


def prepare_data(text, vocab):
    tokens = list(text.lower())
    encoded = torch.tensor([vocab[char] for char in tokens])
    return encoded.unsqueeze(1)

def train_model(text, vocab, model, num_epochs=10, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    encoded_text = prepare_data(text,vocab)
    target = torch.randint(0, len(vocab), (1,)).long()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(encoded_text)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return model


text = "mempertanggungjawabkan adalah kewajiban setiap warga negara."
vocab = {' ': 0, 'a': 1, 'b': 2, 'd': 3, 'e': 4, 'g': 5, 'h': 6, 'i': 7, 'j': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'p': 13, 'r': 14, 's': 15, 't': 16, 'u': 17, 'w': 18, '.': 19}

model = CharLevelLSTM(len(vocab), 50, 100, len(vocab))
trained_model = train_model(text,vocab, model, num_epochs=500)
# For actual prediction, the output logits will need to be converted to actual characters
# Note: this is a highly simplistic example and is not representative of actual training process
# but instead shows that the model learns on characters rather than whole words

```

This illustrates a simplistic training of a character-level lstm, where the character-level tokens of the training text is given as input. A more elaborate process can be designed to make it better suited for a word-level task, for instance, predicting the next word in a sequence. The primary advantage of this character-level approach is that it completely circumvents the complex morphological rules of Indonesian, and learns the representations of characters within words, and combinations of characters representing complex or compound tokens.

For further exploration, I would recommend looking into the *“Speech and Language Processing”* by Daniel Jurafsky and James H. Martin. Also, research papers on subword tokenization like BPE and Wordpiece would prove invaluable for getting a deep understanding. For more detailed study of Indonesian morphology, *“The Morphology of Indonesian”* by M. S. Abdullah could be a good starting point. The *transformers* library documentation from HuggingFace is a great resource for actually implementing the models on large datasets.

In my experience, a combination of these techniques yields the best results. Start with a good rule-based system, then supplement it with a statistical method like BPE, and depending on the project goals, consider character-level representation. The right approach will depend largely on the project requirements, the size of the data, and the required accuracy level.
