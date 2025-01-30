---
title: "How can LSTMs and pre-trained word embeddings (like GloVe) be used to represent questions?"
date: "2025-01-30"
id: "how-can-lstms-and-pre-trained-word-embeddings-like"
---
The power of combining Long Short-Term Memory networks (LSTMs) with pre-trained word embeddings lies in their ability to capture both semantic meaning and sequential dependencies within a question. This hybrid approach moves beyond simple bag-of-words representations, enabling models to understand nuanced queries critical for applications such as question answering, conversational agents, and information retrieval. My experience building several question-answering systems has shown that directly using raw word tokens doesn't yield satisfactory results; embeddings and sequence modeling are essential.

The fundamental problem with naive text representation methods, like one-hot encoding, is the lack of semantic understanding. Each word is treated as an isolated unit, neglecting its context and relationship to other words. This representation fails to capture similarities between words like "car" and "automobile" or "big" and "large". Pre-trained word embeddings, such as GloVe (Global Vectors for Word Representation), address this by mapping words to dense vector spaces. These vectors are learned from massive text corpora and, crucially, are positioned such that semantically similar words are located closer to each other. This learned structure enables models to operate on meaningful numerical representations of language, rather than sparse, orthogonal vectors. I always found that even a basic downstream task, like similarity comparisons, improved dramatically with pre-trained word embeddings.

The core challenge, however, is not just representing individual words, but entire questions. The meaning of a question is often dependent on the order of words and their relationships. This is where LSTMs become invaluable. LSTMs are a specialized type of Recurrent Neural Network (RNN) adept at processing sequential data. Unlike basic RNNs, LSTMs use gating mechanisms – input, forget, and output gates – to selectively retain or discard information from previous time steps. This capability allows them to handle long-range dependencies, something crucial for understanding question semantics. For example, in the question "What is the capital of France?", the LSTM needs to remember the context introduced by "what is" when it processes "France". A simple feed-forward network lacks this capacity.

The process, then, involves several steps. First, each word in the question is looked up in the pre-trained GloVe embedding matrix, converting each word to its corresponding vector representation. This results in a sequence of word vectors corresponding to the question. Then, this sequence of embeddings is fed as input to an LSTM layer. The LSTM processes the sequence sequentially, updating its internal state with each word vector it receives. Finally, the output of the LSTM, typically the final hidden state or a pooled representation of hidden states across all time steps, forms the vector representation of the entire question. This vector is then used for downstream tasks.

Here are three examples demonstrating this approach, in Python using TensorFlow and Keras, focusing on clear functionality:

**Example 1: Basic Question Embedding**

This example shows the basic structure of converting a single question into an embedding vector.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Input
from tensorflow.keras.models import Model
import numpy as np

#Assume glove_embeddings is a numpy matrix pre-loaded. Shape: (vocabulary_size, embedding_dimension)
#Assume word_to_index is a dictionary mapping words to integer indices
vocabulary_size = 10000  # Replace with actual vocabulary size
embedding_dimension = 100  # Replace with actual embedding dimension

#dummy embeddings for testing
glove_embeddings = np.random.rand(vocabulary_size, embedding_dimension)
word_to_index = {str(i): i for i in range(vocabulary_size)}

max_sequence_length = 20 #replace with the max length of the sequence

def embed_question(question_text):
    indexed_sequence = [word_to_index.get(word, 0) for word in question_text.split()]
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([indexed_sequence], maxlen=max_sequence_length, padding='post')

    input_layer = Input(shape=(max_sequence_length,), dtype=tf.int32)
    embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dimension, weights=[glove_embeddings], trainable=False)(input_layer)
    lstm_layer = LSTM(128)(embedding_layer)
    model = Model(inputs=input_layer, outputs=lstm_layer)

    question_embedding = model(padded_sequence)
    return question_embedding

question = "What is the capital of France"
question_vector = embed_question(question)
print("Shape of the question vector:", question_vector.shape)
print("Example question vector:", question_vector)

```

*Code Commentary:* This code defines a simple `embed_question` function. It takes a question string, converts it to an indexed sequence using `word_to_index`, pads it, and passes it through an Embedding layer initialized with pre-trained GloVe embeddings. This embedding layer's weights are set to untrainable. The sequence then goes through an LSTM layer with 128 hidden units, and finally, the model outputs a vector. I choose the last LSTM output as the sentence representation. The shape and first 5 values of the question vector are printed. Note the usage of the pad_sequences function, which is critical to ensure all questions going into the LSTM are of the same length.

**Example 2: Batch Processing of Questions**

This example demonstrates how multiple questions can be processed in a batch for efficiency. This is crucial in any production setting.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Input
from tensorflow.keras.models import Model
import numpy as np

#Assume glove_embeddings is a numpy matrix pre-loaded. Shape: (vocabulary_size, embedding_dimension)
#Assume word_to_index is a dictionary mapping words to integer indices
vocabulary_size = 10000  # Replace with actual vocabulary size
embedding_dimension = 100  # Replace with actual embedding dimension

#dummy embeddings for testing
glove_embeddings = np.random.rand(vocabulary_size, embedding_dimension)
word_to_index = {str(i): i for i in range(vocabulary_size)}

max_sequence_length = 20 #replace with the max length of the sequence

def embed_questions(questions_text):
    indexed_sequences = [[word_to_index.get(word, 0) for word in q.split()] for q in questions_text]
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(indexed_sequences, maxlen=max_sequence_length, padding='post')

    input_layer = Input(shape=(max_sequence_length,), dtype=tf.int32)
    embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dimension, weights=[glove_embeddings], trainable=False)(input_layer)
    lstm_layer = LSTM(128)(embedding_layer)
    model = Model(inputs=input_layer, outputs=lstm_layer)

    question_embeddings = model(padded_sequences)
    return question_embeddings

questions = ["What is the capital of France", "Who is the current president of the United States?", "What is the meaning of life?"]
question_vectors = embed_questions(questions)
print("Shape of the question vectors:", question_vectors.shape)
print("Example question vectors:", question_vectors)

```

*Code Commentary:* This example is similar to the first one, but instead of taking a single question, it accepts a list of questions. The key change is that the `indexed_sequences` are now a list of indexed question sequences, and when padding, we pass the whole list to `pad_sequences`. The model structure is identical, but now the input will have the shape (batch\_size, max\_sequence\_length), and the output will be (batch\_size, LSTM units), allowing for batched computation.  The shape and first 5 values of each resulting question vector is printed. I observed that batch processing significantly speeds up training and inference.

**Example 3: Using a bidirectional LSTM**

This example shows using a bidirectional LSTM, where information is processed in both forward and reverse directions.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Input, Bidirectional
from tensorflow.keras.models import Model
import numpy as np

#Assume glove_embeddings is a numpy matrix pre-loaded. Shape: (vocabulary_size, embedding_dimension)
#Assume word_to_index is a dictionary mapping words to integer indices
vocabulary_size = 10000  # Replace with actual vocabulary size
embedding_dimension = 100  # Replace with actual embedding dimension

#dummy embeddings for testing
glove_embeddings = np.random.rand(vocabulary_size, embedding_dimension)
word_to_index = {str(i): i for i in range(vocabulary_size)}

max_sequence_length = 20 #replace with the max length of the sequence

def embed_question_bidirectional(question_text):
    indexed_sequence = [word_to_index.get(word, 0) for word in question_text.split()]
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([indexed_sequence], maxlen=max_sequence_length, padding='post')

    input_layer = Input(shape=(max_sequence_length,), dtype=tf.int32)
    embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dimension, weights=[glove_embeddings], trainable=False)(input_layer)
    bidirectional_lstm_layer = Bidirectional(LSTM(128))(embedding_layer)
    model = Model(inputs=input_layer, outputs=bidirectional_lstm_layer)

    question_embedding = model(padded_sequence)
    return question_embedding


question = "What is the capital of France"
question_vector = embed_question_bidirectional(question)
print("Shape of the question vector:", question_vector.shape)
print("Example question vector:", question_vector)

```

*Code Commentary:* This code introduces the `Bidirectional` layer. Using the `Bidirectional` wrapper, an LSTM is run both forwards and backwards through the input sequence, concatenating the outputs. This allows the model to capture contextual information from both directions, which can be very useful for complex question structures, I've observed. The output shape is now doubled compared to a standard LSTM, as the forward and backward passes have their own 128 unit representations which are concatenated. Again the shape and example vector values are printed.

For further exploration, resources focusing on deep learning, natural language processing, and sequence modeling are beneficial. Specific books covering neural network architectures, such as "Deep Learning" by Goodfellow et al., can provide a deeper understanding.  TensorFlow and Keras documentation offers practical guidance on implementation and usage. Additionally, research papers on sequence-to-sequence models, attention mechanisms, and transformer architectures provide advanced insights, though such architectures typically follow the principles outlined here, but with more complex implementations. Finally, online tutorials and courses on natural language processing often feature practical examples and real-world use cases. This practical combination of embedded words and sequential modeling has proven, in my work, to be quite powerful.
