---
title: "Why does TensorFlow Keras's `text_to_sequence` return a list of lists?"
date: "2025-01-30"
id: "why-does-tensorflow-kerass-texttosequence-return-a-list"
---
The TensorFlow Keras `text_to_sequences` method, used in text preprocessing, returns a list of lists because it's designed to handle multiple input texts simultaneously, each of which may have a varying number of tokens after tokenization. This architecture enables efficient batch processing common in deep learning applications.

My experience developing a sentiment analysis model for customer reviews highlighted the necessity of this format. Initially, I expected a single list of integer sequences when working with a list of text inputs. However, the nested structure proved essential for batching during training. Each inner list represents the sequence of integer-encoded tokens corresponding to a single input text, while the outer list groups these sequences. This structure is fundamentally different from single text processing and caters to the nature of neural network training, which often benefits from parallel computation across multiple data points (batches).

The Keras `Tokenizer` class, which `text_to_sequences` is a method of, manages this transformation. After fitting the tokenizer on training text (using the `fit_on_texts` method), it constructs a vocabulary mapping each unique word to a numerical index. This mapping is stored internally. When `text_to_sequences` is called, it uses this vocabulary to convert each text into a sequence of integers, where each integer represents a word. Since we typically pass a list of text inputs, we receive a list of corresponding lists, each with the integer sequence from one of our input texts.

Here are some concrete examples demonstrating the output:

**Example 1: Basic Tokenization of Multiple Texts**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

texts = [
    "This is the first sentence.",
    "Here is another sentence, which is slightly longer.",
    "A third very short sentence."
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
print(sequences)
```

**Commentary:**
In this example, we have three different strings stored in the `texts` list. The `Tokenizer` is initialized, and `fit_on_texts` generates the vocabulary based on words in these texts. The subsequent call to `texts_to_sequences` transforms the input `texts` into a list of lists. The output will look like `[[2, 3, 1, 4, 5], [6, 3, 7, 5, 8, 3, 9, 10], [11, 12, 13, 14, 5]]`, assuming some arbitrary word indices were assigned during `fit_on_texts`. Observe that each element of the outer list represents a tokenized sequence derived from each individual string within the initial `texts` list. Crucially, each sequence has a variable length, aligned with the length of its source text.

**Example 2: Handling Out-of-Vocabulary (OOV) Words**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

texts = [
    "This is the first sentence.",
    "Here is a brand new word in this sentence.",
    "This is the third short sentence."
]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
print(sequences)

new_texts = ["Another completely new phrase here."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
print(new_sequences)
```
**Commentary:**
Here, I introduce the `oov_token` parameter to the `Tokenizer`. This handles words not seen during `fit_on_texts`. In the `texts`, "brand," "new," and "word" are used for the first time. Upon converting those three strings to sequences, the OOV token will be used for "brand" and "word" since it does not exist in the vocabulary generated from all 3 strings in the `texts` list. The output for `sequences` could be like this: `[[2, 3, 1, 4, 5], [6, 3, 7, 8, 9, 10, 11, 12], [2, 3, 13, 14, 5]]`, assuming again that some arbitrary numbers have been assigned. Subsequently, a completely new text `new_texts` containing only out-of-vocabulary words is provided. The result in `new_sequences` becomes `[[1, 1, 1, 1]]` since all words in the `new_texts` list are unknown, and the `<OOV>` token is assigned to them instead. This demonstrates that, during the sequence creation process, each sentence is processed individually while also following the provided OOV token rule.

**Example 3: Padding for Batching**
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    "This is a short sentence.",
    "This is a sentence longer than the previous one, quite a bit longer.",
    "Another short one."
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(sequences, padding='post')
print(padded_sequences)
```
**Commentary:**
This example demonstrates how the nested list structure generated by `text_to_sequences` is immediately useful for subsequent batch-processing steps. After tokenizing the texts, the variable-length sequences are padded to have a uniform length using `pad_sequences`. By default, sequences are padded with zeros at the beginning to the length of the longest sequence, or according to the length specified by `maxlen`.  In this case, post-padding is selected, which adds zeros at the end. The output `padded_sequences` becomes a NumPy array, which can be directly fed into a Keras neural network model. For instance, the output could be something like: `[[ 2  3  4  5  6  0  0  0  0  0] [ 2  3  4  5  7  8  9 10 11 12] [13 14 15  0  0  0  0  0  0  0]]`. Notice how, for training, the variable lengths from the initial list of lists are resolved with the uniform padding. This step is not possible without the initial nested list output of `text_to_sequences`.

Based on these experiences, I can state definitively that the nested list structure returned by Keras' `text_to_sequences` is not a design quirk, but rather an essential feature for managing batches of variable-length texts that are a common input for natural language processing models. The structure enables subsequent preprocessing steps like padding, which is required for tensor operations during training.

To gain a deeper understanding of this, I highly recommend consulting the Keras documentation on text preprocessing and specifically the `Tokenizer` class. Additionally, the official TensorFlow tutorials on NLP provide practical examples showcasing this workflow. A thorough review of the academic literature on sequence models, such as RNNs and Transformers, will also clarify why variable-length sequences need to be converted into uniform length for efficient batch processing. The key areas to focus on are batch processing, tokenization, sequence padding, and OOV management. Exploring these concepts will provide a holistic understanding of the method's rationale. Furthermore, examining code examples within various open-source NLP libraries demonstrates that these preprocessing steps are common and the nested list structure is a widely used approach.
