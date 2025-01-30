---
title: "How can I combine BERT sentence embeddings and word embeddings using Keras and Hugging Face?"
date: "2025-01-30"
id: "how-can-i-combine-bert-sentence-embeddings-and"
---
The challenge in effectively combining BERT sentence embeddings and word embeddings arises from their differing scopes; BERT's output represents a contextualized understanding of an entire sentence, while word embeddings provide granular, token-level semantic representations. Bridging this gap requires careful consideration of how the information at these distinct levels can be integrated to benefit a downstream task. From personal experience working on a complex document summarization project, I found that treating these embeddings as separate, yet complementary, input channels proved effective, rather than attempting a direct mathematical fusion. Here’s how this can be achieved using Keras and Hugging Face Transformers.

**Understanding the Problem**

BERT, as implemented through Hugging Face's `transformers` library, outputs several types of embeddings. The `[CLS]` token embedding is typically taken as a representation of the entire input sequence, functioning as the sentence embedding. Conversely, the embedding corresponding to each token in the input provides word-level representations. My initial assumption was a simple concatenation would suffice, but experiments proved this too simplistic; it failed to capture the nuanced interdependencies between the sentence context and individual word meanings. Consequently, I moved towards architectures that process these embeddings separately and then fuse them through learned mechanisms. This is key - the raw embeddings shouldn’t be forced into a single vector early.

**A Dual-Path Approach**

My proposed solution uses a dual-path architecture within Keras. One path takes the BERT sentence embeddings (the `[CLS]` token) as input, while the other processes the sequence of BERT word embeddings. This separation allows each type of information to be processed by layers designed to suit their scale and content. The outputs of these parallel paths are then concatenated before feeding into a final predictive or classification layer. I'll illustrate with Keras and Hugging Face library elements, using a hypothetical multi-class text classification problem.

**Code Examples and Explanations**

*   **Example 1: Setting Up the Input and BERT Model**

    This initial setup focuses on loading a pre-trained BERT model and preparing data inputs. This is the common initial phase, before the parallel processing is implemented.

    ```python
    import tensorflow as tf
    from transformers import TFBertModel, BertTokenizer
    import numpy as np

    # Load BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = TFBertModel.from_pretrained(model_name)

    # Example data input (replace with your actual data)
    sentences = ["This is the first sentence.", "And this is the second one."]
    labels = [0, 1] # Example labels for a two-class problem

    # Tokenize input sequences
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='tf')

    # Function to get BERT sentence and word embeddings
    def get_bert_embeddings(encoded_inputs, model):
        outputs = model(encoded_inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        word_embeddings = outputs.last_hidden_state        # Embeddings for all words, including the CLS
        return sentence_embeddings, word_embeddings

    sentence_embeddings, word_embeddings = get_bert_embeddings(encoded_inputs, bert_model)
    print("Sentence embeddings shape:", sentence_embeddings.shape) # Expected shape: (batch_size, hidden_size)
    print("Word embeddings shape:", word_embeddings.shape) # Expected shape: (batch_size, sequence_length, hidden_size)
    ```

    This code segment uses Hugging Face's `TFBertModel` and `BertTokenizer` to load a pre-trained BERT model. The `tokenizer` converts input sentences into a format usable by BERT, with padding and truncation to ensure uniform sequence lengths. The `get_bert_embeddings` function extracts both the `[CLS]` token embedding, as a representation of the full sentence, and the complete sequence of token embeddings, allowing us to then differentiate processing paths. Observe the differing shapes of the resulting embeddings - one represents the sentence and the other represents each token.
*   **Example 2: Building the Dual-Path Keras Model**

    This example constructs the Keras model incorporating both types of BERT embeddings.

    ```python
    from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Layer
    from tensorflow.keras.models import Model
    
    # Define input shapes
    sentence_embedding_shape = sentence_embeddings.shape[-1]
    word_embedding_shape = word_embeddings.shape[-1]
    sequence_length = word_embeddings.shape[1]


    # Input layers
    sentence_input = Input(shape=(sentence_embedding_shape,), name='sentence_input')
    word_input = Input(shape=(sequence_length, word_embedding_shape), name='word_input')

    # Sentence embedding path
    sentence_dense = Dense(128, activation='relu', name='sentence_dense')(sentence_input)

    # Word embedding path - an LSTM to process the sequence of word vectors.
    word_lstm = LSTM(128, name='word_lstm')(word_input)

    # Concatenate the processed embeddings
    merged = Concatenate()([sentence_dense, word_lstm])

    # Output layer for classification
    output = Dense(2, activation='softmax', name='output')(merged) # Assuming a 2-class classification

    # Create the model
    model = Model(inputs=[sentence_input, word_input], outputs=output)

    # Model summary
    model.summary()

    # Compile the model (example)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Prepare input for the Keras model - convert to NumPy for TF 2.x compatibility
    sentence_input_data = sentence_embeddings.numpy()
    word_input_data = word_embeddings.numpy()
    labels = np.array(labels)

    # Train the model (example)
    model.fit([sentence_input_data, word_input_data], labels, epochs=5, batch_size=2)
    ```

    This code shows how the dual-path model architecture can be implemented using the Keras functional API. Separate input layers are defined for the sentence embeddings and the sequence of word embeddings. The sentence embeddings go through a dense layer. The word embeddings are fed into an LSTM layer, which processes the sequence and generates a single vector representation of the entire word sequence. These outputs are concatenated, creating a single feature vector. A final dense layer outputs class probabilities. The `model.summary()` gives a clear view of the architecture, and provides the layers in a concise manner.
*   **Example 3: Custom Layer for Word Embedding Pooling**

    While the LSTM is effective, alternative pooling methods can be more computationally efficient. This example demonstrates a custom pooling layer, which can be used in place of the LSTM.

    ```python
    from tensorflow.keras.layers import Layer
    import tensorflow as tf
    # Custom Pooling Layer
    class MeanPoolingLayer(Layer):
       def __init__(self, **kwargs):
           super(MeanPoolingLayer, self).__init__(**kwargs)

       def call(self, inputs):
          return tf.reduce_mean(inputs, axis=1) # Mean pooling over the sequence length

       def compute_output_shape(self, input_shape):
          return (input_shape[0], input_shape[2])
    # Input layers
    sentence_input_alt = Input(shape=(sentence_embedding_shape,), name='sentence_input_alt')
    word_input_alt = Input(shape=(sequence_length, word_embedding_shape), name='word_input_alt')
    
    # Sentence embedding path
    sentence_dense_alt = Dense(128, activation='relu', name='sentence_dense_alt')(sentence_input_alt)

    # Word embedding path - mean pooling for the word vectors
    word_pooling_alt = MeanPoolingLayer()(word_input_alt)
    
    # Concatenate the processed embeddings
    merged_alt = Concatenate()([sentence_dense_alt, word_pooling_alt])

    # Output layer for classification
    output_alt = Dense(2, activation='softmax', name='output_alt')(merged_alt) # Assuming a 2-class classification

    # Create the model
    model_alt = Model(inputs=[sentence_input_alt, word_input_alt], outputs=output_alt)

    # Model summary
    model_alt.summary()
    # Compile and Train the model (as in previous example)
    model_alt.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model (example)
    model_alt.fit([sentence_input_data, word_input_data], labels, epochs=5, batch_size=2)
    ```

    This snippet introduces `MeanPoolingLayer`, a custom layer to perform mean pooling over the sequence of word embeddings. Mean pooling involves averaging the vectors across the sequence length. This is computationally less expensive than the LSTM, and may be sufficient for specific tasks. Using the custom `MeanPoolingLayer` simplifies integration; a `compute_output_shape` method is needed to ensure dimension compatibility with subsequent layers.

**Considerations and Resource Recommendations**

The selection of the most appropriate model is dependent on the downstream task. If capturing long-range dependencies between words within the sentence is crucial, the LSTM will generally outperform a mean pooling strategy. However, mean pooling will likely yield quicker training times, and potentially more robust solutions. These considerations are dependent on the data being processed. The architecture demonstrated here can be modified, including more complex layers for processing each embedding type, multiple hidden layers, attention mechanisms, and customized loss functions.

For further study on working with language models and Keras, I suggest the following resources:

1.  **The official TensorFlow documentation**: Provides a comprehensive understanding of Keras layers, model building, and training processes.
2.  **The Hugging Face Transformers documentation**: Offers thorough explanations of their pre-trained models, tokenizers, and utilities, crucial for effective utilization of BERT and similar models.
3.  **Online machine learning courses**: Platforms like Coursera and edX offer courses covering natural language processing with deep learning techniques, which are often built using Keras and TensorFlow.

My experience highlighted the importance of not just blindly combining embeddings, but building a thoughtful architecture that allows each type of representation to contribute optimally. This approach, with the provided examples, should serve as a foundation for your task.
