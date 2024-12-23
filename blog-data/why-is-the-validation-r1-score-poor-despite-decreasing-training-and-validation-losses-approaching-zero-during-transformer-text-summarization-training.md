---
title: "Why is the validation R1 score poor despite decreasing training and validation losses approaching zero during transformer text summarization training?"
date: "2024-12-23"
id: "why-is-the-validation-r1-score-poor-despite-decreasing-training-and-validation-losses-approaching-zero-during-transformer-text-summarization-training"
---

Alright, let’s tackle this one. I’ve definitely seen this particular flavor of frustration before, particularly back when I was knee-deep in developing a text summarization system for a legal documentation platform. Witnessing training and validation losses plummet, only to be slapped in the face with a lackluster R1 score, can be quite perplexing. It essentially boils down to the nuances of loss function optimization versus actual summarization quality, and it’s a common pitfall.

The core issue here is that loss functions, particularly those used in training transformers like cross-entropy, focus on accurately predicting the next token in a sequence. They’re designed to minimize the difference between the predicted token probability distribution and the actual target token distribution. While a decreasing loss indicates our model is improving at this prediction task, it’s not directly equivalent to producing good *summaries*. Think of it as becoming incredibly good at reciting a textbook word-for-word, but without necessarily understanding the core concepts or being able to generate a concise summary of the key ideas.

R1 score, in contrast, evaluates the overlap of n-grams (usually unigrams, bigrams, and trigrams) between the generated summary and the reference summary. This is a surface-level metric that rewards word-level matching, not conceptual understanding or semantic coherence. A model might learn to perfectly recreate chunks of the original text, or learn to generate summaries that are syntactically similar, while missing the essence of a good summary—namely, brevity, relevance, and meaning. That perfectly recreated chunk of the original text would get a great R1 score if it is part of the reference summary, however it won’t actually create a true summary. This is essentially the discrepancy between learning to copy and learning to truly summarize.

Let’s break down some reasons why this happens with transformer-based text summarization models:

1.  **Optimization on Token Prediction, Not Summary Quality:** As I previously touched upon, the training process primarily focuses on reducing token prediction loss. Cross-entropy pushes the model to match the distribution of predicted tokens with the actual tokens in the target summary. If your training data contains repetitive structures, or the model is overly influenced by frequency of particular n-grams, it may produce highly repetitive or uninformative summaries that match the reference on a word level, thus improving R1, but may not be an adequate summary. The model can excel at this localized token matching without grasping the overall meaning.

2.  **N-gram Matching Limitations:** R1 score inherently has limitations. It does not take into account synonyms, paraphrasing, or semantic meaning. Two summaries can express the same idea with different vocabulary and sentence structure, and one may get penalized if the word-level matching is poor. R1 score only really values exact matching of word sequences. A model could theoretically write a terrible summary with low semantic cohesion, but get a respectable R1 score because many of the words are present in the right sequence.

3. **Decoder Issues:** The decoder component of the transformer could also be generating summaries that are syntactically correct but semantically incoherent. A high R1 might occur if the model is learning to generate templates or stereotypical summary structures while only filling them with words it saw in the reference text. You might see it generating summaries that look like a list of facts that are directly copied from the input text rather than distilled.

4.  **Training Data Bias:** Another possibility is the training data itself. If your training data contains more data that is of similar structure, or contains many similar summaries, the model could be learning to mimic these patterns rather than generalize towards good summarization. This includes things such as an over reliance on direct quotations. The model might be optimizing for R1 in the training set, which doesn't generalize outside the training dataset.

Let's look at some code snippets to illustrate these points. The code will be simplified for illustrative purposes, but should convey the ideas. First, we’ll create a basic summarization scoring method based on overlap:

```python
from collections import Counter

def r1_score(summary, reference):
    summary_words = summary.lower().split()
    reference_words = reference.lower().split()

    summary_counts = Counter(summary_words)
    reference_counts = Counter(reference_words)

    overlap = sum((summary_counts & reference_counts).values())
    total_words = sum(reference_counts.values())

    if total_words == 0:
      return 0
    return overlap / total_words

example_summary_1 = "The cat sat on the mat."
example_reference_1 = "The feline was located upon the rug."

example_summary_2 = "The cat sat on the mat. The mat was green."
example_reference_2 = "The cat sat on the mat."

r1_score_1 = r1_score(example_summary_1, example_reference_1)
r1_score_2 = r1_score(example_summary_2, example_reference_2)

print(f"R1 score example 1: {r1_score_1}")
print(f"R1 score example 2: {r1_score_2}")

```

The first example shows how the score is penalized for using different words, even if the core idea is similar (synonyms), which shows a basic limitation of the score. The second example illustrates how a summary can achieve a perfect score by being an expansion of a reference, but not a true summary. The same word order is heavily favored.

Now, let's say we train a transformer model with a simple cross-entropy loss. This is overly simplified, but the core idea is the same:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleTransformer(nn.Module): # overly simplified for demonstration
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.Linear(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        encoded = self.encoder(embedded)
        output = self.decoder(encoded)
        return output

vocab_size = 100
embedding_dim = 32
hidden_dim = 64

model = SimpleTransformer(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

input_sequence = torch.randint(0, vocab_size, (1, 10)) # simplified input
target_sequence = torch.randint(0, vocab_size, (1, 10)) # simplified target

for epoch in range(100):
    optimizer.zero_grad()
    output = model(input_sequence)
    loss = criterion(output.permute(0,2,1), target_sequence)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


```

This simplified training loop focuses entirely on loss reduction using cross-entropy, without any explicit attention to summarization quality metrics. The model learns to predict the next token from its input based on the *local* context without regard to the quality of a concise summary.

Finally, consider a scenario where a model generates summaries by directly copying portions of the input sequence. This often happens because the embedding will push similar word patterns closer together in the higher dimensions, and the model will learn to identify these common patterns and repeat them without any real understanding:

```python
def generate_summary_copy(input_text, summary_length):
  words = input_text.split()
  return " ".join(words[:summary_length])

input_text_example = "This is a long document with many words that need to be summarized to a concise statement for easy consumption."
generated_summary_example = generate_summary_copy(input_text_example, 10)
reference_summary_example = "Long document needs concise summary for easy consumption."

r1_score_copy = r1_score(generated_summary_example, reference_summary_example)
print(f"Generated summary by copy: {generated_summary_example}")
print(f"R1 score by copy: {r1_score_copy}")
```

In this last code segment, the copy method of summarizing generates a summary that will score higher if the reference is similar to the start of the input text, despite not actually accomplishing summarization.

So, what can be done? Firstly, focus on training metrics that directly correlate with summarization performance. ROUGE scores, which include R1, R2, and RL, are a standard in summarization but they still suffer from many of the issues discussed earlier. You may also want to explore metrics like BERTScore, which evaluates semantic similarity.

Secondly, consider different training strategies. Reinforcement learning (RL), specifically policies that maximize summarization metrics (like ROUGE or BERTScore) can be effective at directly optimizing for summary quality, rather than relying on cross-entropy as a proxy. Additionally, you can explore training on larger and more diverse datasets. In the original legal documentation project I worked on, using a mixture of human-generated summaries and carefully curated synthetic data had a noticeable positive impact. Techniques such as pointer networks, which allow the model to copy directly from the input text when appropriate, or training with noise and perturbation can also produce better results.

Lastly, and perhaps most importantly, don't rely solely on automated metrics. It is essential to have human evaluations of the summaries to fully understand how well the model has truly grasped the material.

For further reading, I highly recommend diving into the *Attention is All You Need* paper (Vaswani et al., 2017), which introduced transformers. To understand evaluation metrics more deeply, look at papers covering the ROUGE metric (*ROUGE: A Package for Automatic Evaluation of Summaries* by Lin, 2004) and BERTScore (*BERTScore: Evaluating Text Generation with BERT* by Zhang et al., 2019). These papers, in addition to others, will provide a robust understanding of the challenges and solutions to the problem of evaluating summaries. It's a challenging area, but with a good understanding of the underlying dynamics, you can definitely make substantial progress towards better summarization systems.
