---
title: "How can I obtain word frequencies using TensorFlow 2's TextVectorization layer?"
date: "2025-01-30"
id: "how-can-i-obtain-word-frequencies-using-tensorflow"
---
The `TextVectorization` layer in TensorFlow 2 offers more than just text-to-integer conversion; it inherently computes term frequencies crucial for tasks like building simple bag-of-words models. Instead of performing a separate counting operation after vectorization, these frequencies are accessible via the layer’s vocabulary, although their format requires understanding. Having personally used this for numerous NLP preprocessing pipelines, I’ve found it essential to extract these frequencies efficiently without redundancy.

The core function of `TextVectorization` is to transform a batch of raw text into a batch of integer sequences, where each unique word (or token) is represented by a unique integer index. By default, it constructs its vocabulary based on the provided text corpus. This vocabulary is not just a simple mapping of words to IDs; it includes additional information, particularly the count of each token. This count information is what we’re interested in to derive word frequencies. The frequencies themselves aren't directly stored as a separate entity but are used during the vocabulary building phase to determine the most frequent terms if the `max_tokens` argument is provided. Lower counts might be ignored based on this limit.

Here's how the process works conceptually:
1. **Initialization:** The `TextVectorization` layer is instantiated with specific configuration parameters such as `max_tokens` and `output_mode`. Crucially, `output_mode` must be set to `"int"` to enable retrieval of token counts.
2. **Adaptation:** The layer's internal state is built by calling `adapt` with the training text corpus. During this process, the layer scans the provided text, tokenizes it (usually by splitting on spaces and removing punctuation), and constructs the vocabulary. Each time it encounters a token it maintains a count.
3. **Vocabulary Access:** The vocabulary, once constructed, can be accessed using the `get_vocabulary()` method. This returns a Python list of tokens, ordered by decreasing frequency, where the first element is the vocabulary’s OOV (out-of-vocabulary) token (usually an empty string) and the last tokens are those that appear least frequently, or might be ignored if a maximum token limit was set.
4. **Frequency Access:** Directly obtaining the raw counts requires accessing the layer’s underlying variable that is generated internally and is not intended for the end user to directly modify. For this we can use `get_config()` to get access to the layer configuration. For example, it contains a variable named `idf_weights` (inverse document frequency) if the output mode is tf-idf. The raw counts of words appear to be implicitly stored in the `adapt` process but are not exposed directly as an end user API. The `idf_weights` or other layer-specific weights are computed using these counts.

Given this indirect access methodology, obtaining the counts require manual computation. After using `get_vocabulary()` to obtain the vocabulary, the text corpus can then be processed via the layer and a simple counting of each output integer mapping to its respective word can be performed. This requires iterating the output of the `TextVectorization` layer on the corpus, and will result in a dictionary of word-counts which then can be converted to frequencies by dividing by the sum of all counts.

Let's illustrate with examples:

**Example 1: Basic Frequency Extraction**

```python
import tensorflow as tf
import numpy as np

# Sample training data
texts = [
  "the quick brown fox jumps over the lazy dog",
  "the cat is sleeping under the table",
  "a big red fox runs fast",
  "dogs love to chase the ball"
]

# Instantiate TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(output_mode="int")

# Adapt the layer to the training texts
vectorizer.adapt(texts)

# Get vocabulary
vocabulary = vectorizer.get_vocabulary()

# Initialize count dictionary
word_counts = {word: 0 for word in vocabulary}

# Count word occurrences by vectorizing each text then updating the count dictionary
for text in texts:
    vectorized_text = vectorizer(np.array([text]))[0] # Vectorize a single text and un-batch
    for token_id in vectorized_text:
      word = vocabulary[token_id]
      if word != '': # Skip OOV index
        word_counts[word] += 1

# Calculate total words
total_words = sum(word_counts.values())

# Calculate frequencies
word_frequencies = {word: count / total_words for word, count in word_counts.items() if count > 0}

print("Word Frequencies:")
for word, freq in sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True):
  print(f"{word}: {freq:.4f}")
```
This example creates a `TextVectorization` layer, adapts it to the sample text, retrieves the vocabulary, and then manually counts the occurrences of each word using the vectorized outputs. The resulting `word_frequencies` dictionary holds the calculated relative frequency for each word in the training data.

**Example 2: Handling Multiple Batches**

In a practical scenario, input texts are processed in batches. The following example shows how to aggregate word counts across multiple batches:

```python
import tensorflow as tf
import numpy as np

# Sample training data
texts = [
  "the quick brown fox jumps over the lazy dog",
  "the cat is sleeping under the table",
  "a big red fox runs fast",
  "dogs love to chase the ball",
  "the sun is shining brightly today",
  "a happy cat is purring",
  "the small dog wags its tail",
  "birds fly high in the sky"
]

# Instantiate TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(output_mode="int")

# Adapt the layer
vectorizer.adapt(texts)

# Get vocabulary
vocabulary = vectorizer.get_vocabulary()

# Initialize count dictionary
word_counts = {word: 0 for word in vocabulary}

# Convert data to TF dataset
dataset = tf.data.Dataset.from_tensor_slices(texts).batch(2) # Create dataset

# Iterate over dataset, unbatch, vectorize and collect counts
for text_batch in dataset:
  for text in text_batch:
    vectorized_text = vectorizer(np.array([text.numpy().decode('utf-8')]))[0] # Unbatch and decode
    for token_id in vectorized_text:
      word = vocabulary[token_id]
      if word != '':
        word_counts[word] += 1

# Calculate total words
total_words = sum(word_counts.values())

# Calculate frequencies
word_frequencies = {word: count / total_words for word, count in word_counts.items() if count > 0}

print("Word Frequencies:")
for word, freq in sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True):
  print(f"{word}: {freq:.4f}")
```

Here, we use `tf.data.Dataset` to handle text batches. Within each batch, we iterate through each text, vectorize it, and update the word counts accordingly. The key is to convert dataset tensors into strings before applying the vectorizer, and also to vectorize individual texts to get a workable un-batched vectorized sequence.

**Example 3: Incorporating a `max_tokens` limit**
This example shows the behaviour of the `max_tokens` setting. If a `max_tokens` is supplied to the text vectorization layer, then only the `max_tokens -1` most common tokens are kept in the vocabulary, all others are considered to be out-of-vocabulary (OOV) and converted to integer index 0.

```python
import tensorflow as tf
import numpy as np

# Sample training data
texts = [
  "the quick brown fox jumps over the lazy dog",
  "the cat is sleeping under the table",
  "a big red fox runs fast",
  "dogs love to chase the ball",
  "the sun is shining brightly today",
  "a happy cat is purring",
  "the small dog wags its tail",
  "birds fly high in the sky",
  "the quick brown fox jumps over the lazy dog again"
]

# Instantiate TextVectorization layer with a max_tokens
vectorizer = tf.keras.layers.TextVectorization(output_mode="int", max_tokens=10)

# Adapt the layer
vectorizer.adapt(texts)

# Get vocabulary
vocabulary = vectorizer.get_vocabulary()

# Initialize count dictionary
word_counts = {word: 0 for word in vocabulary}

# Convert data to TF dataset
dataset = tf.data.Dataset.from_tensor_slices(texts).batch(2) # Create dataset

# Iterate over dataset, unbatch, vectorize and collect counts
for text_batch in dataset:
  for text in text_batch:
    vectorized_text = vectorizer(np.array([text.numpy().decode('utf-8')]))[0] # Unbatch and decode
    for token_id in vectorized_text:
      word = vocabulary[token_id]
      if word != '':
        word_counts[word] += 1

# Calculate total words
total_words = sum(word_counts.values())

# Calculate frequencies
word_frequencies = {word: count / total_words for word, count in word_counts.items() if count > 0}

print("Word Frequencies:")
for word, freq in sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True):
  print(f"{word}: {freq:.4f}")
```
This code snippet limits the vocabulary to 10 tokens (including the OOV token at index 0) and so only the 9 most frequent words in the input corpus are considered during the frequency counting.

**Resource Recommendations:**
For comprehensive information on `TextVectorization` and related text preprocessing techniques in TensorFlow 2, consulting the official TensorFlow documentation is crucial. Reviewing tutorials and examples provided by the TensorFlow team, particularly those focused on natural language processing, will provide practical application scenarios. The Keras API documentation specifically provides thorough explanations for various layers and their parameters. Furthermore, examining code examples from open-source NLP projects hosted on platforms like GitHub will give exposure to advanced techniques and best practices in text data processing.
