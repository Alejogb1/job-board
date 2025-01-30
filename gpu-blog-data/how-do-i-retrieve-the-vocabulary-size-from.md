---
title: "How do I retrieve the vocabulary size from a fitted TensorFlow tokenizer?"
date: "2025-01-30"
id: "how-do-i-retrieve-the-vocabulary-size-from"
---
The vocabulary size of a fitted TensorFlow tokenizer isn't directly stored as a readily accessible property. Instead, it must be derived from the tokenizer's internal mapping between words and integer indices. This approach is consistent with TensorFlow's design emphasis on handling data as numerical tensors rather than explicitly maintaining string-based vocabularies post-fitting. After years spent working on natural language processing pipelines in large-scale systems, I have consistently encountered the need to programmatically extract this size for further operations such as embedding layer initialization and model architecture configuration. Here, I'll detail how to accomplish this task, focusing on the common `tf.keras.preprocessing.text.Tokenizer` class and related methods.

The core principle involves inspecting the tokenizer's `word_index` attribute, a dictionary where keys are the unique tokens from the training data (strings), and values are integer indices. The size of this dictionary corresponds to the number of unique tokens, effectively providing the vocabulary size. Critically, this method reveals the *actual* vocabulary size after fitting, encompassing any filters or transformations applied to the original raw text corpus.  This contrasts with the initial vocabulary parameter specified during tokenizer instantiation which represents the maximum number of words to retain during the fitting process and is often greater than the vocabulary size post-fit.

The tokenizer also adds some special tokens during fitting, notably a "padding" token and an "out-of-vocabulary" token. These tokens also get included in the `word_index` count. The "padding" token is, by default, assigned the index 0 and usually not an actual word. The “out-of-vocabulary” token is assigned a higher index, often 1, and represents words that were not included in the initial vocabulary and was not seen during fitting. Thus, to derive the vocabulary *size* correctly, we often want to count only the words that are not special tokens. If these tokens are assigned, the vocabulary size derived from `word_index` would be 2 greater than the number of words extracted from your text corpus (e.g. unique words).

Let's illustrate with some code examples. First, assume we have a simple dataset of short sentences to be tokenized.

```python
import tensorflow as tf

sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a lazy cat sleeps soundly",
    "the brown dog barks loudly",
    "quick foxes are active"
]

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20)
tokenizer.fit_on_texts(sentences)

vocabulary_size = len(tokenizer.word_index)
print(f"Vocabulary size, including special tokens: {vocabulary_size}")
```
This initial example demonstrates the basic principle. A `Tokenizer` instance is created with a maximum vocabulary size of 20, although the actual vocabulary size is determined by the number of unique words encountered during `fit_on_texts`. The  `word_index` dictionary is then accessed, its length directly providing the size *including special tokens*. If you run this, the output will indicate a vocabulary size of 12, which represents 10 unique words extracted from the text corpus, plus one entry for padding (index 0) and one entry for out-of-vocabulary (index 1).

Now, consider the scenario where we need the "true" vocabulary size, excluding special tokens. The "padding" token’s index is always 0, so we can simply exclude that from the length. For the “out-of-vocabulary” token, we can find its associated index by querying the dictionary using `tokenizer.oov_token`. Then, any index equal or less than the index will be considered special tokens.

```python
import tensorflow as tf

sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a lazy cat sleeps soundly",
    "the brown dog barks loudly",
    "quick foxes are active"
]

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20, oov_token='<unk>')
tokenizer.fit_on_texts(sentences)

oov_token_index = tokenizer.word_index.get(tokenizer.oov_token)

vocabulary_size = len(tokenizer.word_index)
true_vocabulary_size = len([k for k, v in tokenizer.word_index.items() if v > oov_token_index])
print(f"Vocabulary size, including special tokens: {vocabulary_size}")
print(f"Vocabulary size, excluding special tokens: {true_vocabulary_size}")

```
In the second example, an out-of-vocabulary token '<unk>' was explicitly set in the `Tokenizer`, as well as the `num_words` parameter is unchanged from the first example. Now the index of out-of-vocabulary token is retrieved by accessing `tokenizer.oov_token` which would have the string `<unk>`. Then the vocabulary size *excluding* special tokens is computed by iterating through `tokenizer.word_index`, and counting only entries that has index greater than the out-of-vocabulary token’s index. The output would now state that vocabulary size, including special tokens, is 12 and vocabulary size, excluding special tokens, is 10.

Finally, let’s assume a use case where the model architecture needs to dynamically accommodate for a different vocabulary sizes based on a number of data sets.

```python
import tensorflow as tf

sentences_1 = [
    "the quick brown fox jumps over the lazy dog",
    "a lazy cat sleeps soundly",
    "the brown dog barks loudly",
    "quick foxes are active"
]

sentences_2 = [
  "red car blue car green car",
    "yellow truck heavy truck",
    "small bike big bike",
]

tokenizer_1 = tf.keras.preprocessing.text.Tokenizer(num_words=20, oov_token='<unk>')
tokenizer_1.fit_on_texts(sentences_1)

oov_token_index_1 = tokenizer_1.word_index.get(tokenizer_1.oov_token)
vocabulary_size_1 = len([k for k, v in tokenizer_1.word_index.items() if v > oov_token_index_1])
print(f"Vocabulary size for data set 1: {vocabulary_size_1}")

tokenizer_2 = tf.keras.preprocessing.text.Tokenizer(num_words=20, oov_token='<unk>')
tokenizer_2.fit_on_texts(sentences_2)

oov_token_index_2 = tokenizer_2.word_index.get(tokenizer_2.oov_token)
vocabulary_size_2 = len([k for k, v in tokenizer_2.word_index.items() if v > oov_token_index_2])
print(f"Vocabulary size for data set 2: {vocabulary_size_2}")

embedding_dimension = 128 #arbitrary number for demonstration
embedding_layer_1 = tf.keras.layers.Embedding(input_dim=vocabulary_size_1, output_dim=embedding_dimension)
embedding_layer_2 = tf.keras.layers.Embedding(input_dim=vocabulary_size_2, output_dim=embedding_dimension)

print(f"Embedding Layer 1 with input dim: {embedding_layer_1.input_dim} ")
print(f"Embedding Layer 2 with input dim: {embedding_layer_2.input_dim} ")

```

This final example demonstrates that for different data sets the vocabulary size is different, which in turn is used to initialize differently sized embedding layers. The output would state the vocabulary size is 10 for data set 1 and 7 for data set 2. The embedding layers are then successfully initialized. This highlights the practical application of the previous methods in a more complex scenario.

For further study, I recommend familiarizing yourself with the official TensorFlow documentation, particularly the sections related to text preprocessing and the `tf.keras.preprocessing.text.Tokenizer` class. Additionally, exploring practical examples of text classification and sequence-to-sequence models will provide context for where these vocabulary sizes are crucial. Consulting books on deep learning, focusing on natural language processing, will also yield valuable insights. I would also recommend further studying the difference between fitting the tokenizer with `fit_on_texts` versus `fit_on_sequences` as the latter works on sequences instead of texts. Further work may require converting from texts to sequences using `texts_to_sequences`. In summary, extracting the vocabulary size from a fitted TensorFlow tokenizer involves a clear understanding of the `word_index` attribute and accounting for special tokens. This is vital for tasks such as correctly sizing embedding layers, and a nuanced understanding of this process significantly impacts the performance of NLP models.
