---
title: "How can CNNs and BERT be used with text data?"
date: "2024-12-23"
id: "how-can-cnns-and-bert-be-used-with-text-data"
---

Okay, let’s tackle this one. It’s a question I’ve grappled with extensively, particularly back when I was building a sentiment analysis engine for a large-scale social media platform. We initially leaned heavily on traditional nlp techniques, but the shift to deep learning, specifically cnn and bert, opened up possibilities we hadn't previously imagined.

The core challenge when processing text with neural networks lies in transforming the symbolic nature of language into numerical representations that these models can effectively consume. Unlike images, where pixels provide inherent spatial structure, text requires a pre-processing stage that is both nuanced and crucial to model performance.

Let's begin with convolutional neural networks (cnns). While often associated with image processing, they've demonstrated surprising efficacy with text. The fundamental principle here is that we treat sentences or documents as one-dimensional sequences, and the convolution filters slide across these sequences, searching for patterns. In effect, we’re looking for local features such as word n-grams or short phrases that carry semantic meaning. The key difference, of course, is using 1-dimensional rather than 2-dimensional convolution operations.

My team's early attempts involved naive one-hot encoding, which resulted in sparse input vectors that were difficult for cnn to learn from effectively. We quickly switched to pre-trained word embeddings (word2vec and glove), each word represented by a relatively dense vector based on contextual information gleaned from a massive corpus. This greatly improved results, because now, we could identify words that are similar in meaning based on vector closeness.

Here's a simplified python code using keras/tensorflow to demonstrate text processing with cnn:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample text data
texts = ["this is a good movie", "bad review", "i love this", "not very good"]
labels = [1, 0, 1, 0] # 1 for positive, 0 for negative

# Tokenization
tokenizer = Tokenizer(num_words=100) # Limit to top 100 most frequent words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Model definition
model = keras.Sequential([
    layers.Embedding(input_dim=100, output_dim=32, input_length=max_len), # Embedding layer
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(1, activation='sigmoid') # Output layer (binary classification)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
padded_sequences_np = np.array(padded_sequences)
labels_np = np.array(labels)
model.fit(padded_sequences_np, labels_np, epochs=10, verbose=0)

# prediction
test_text = ["this is a great one"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(test_seq, maxlen=max_len, padding='post')
prediction = model.predict(np.array(test_padded))
print(f"prediction {prediction}")

```

This code snippet illustrates the basic steps: tokenization, padding for uniform input length, embedding, convolution, max pooling, and classification. Note the use of `conv1d` for processing the sequence data. The `globalmaxpooling1d` helps to extract the most important features after the convolutions, while the embedding layer enables us to learn a vector representation of the tokens within our text data.

Now, moving on to bert (bidirectional encoder representations from transformers), this is where things get truly interesting. Bert, unlike traditional models, leverages the transformer architecture and processes input text bidirectionally. This means it considers both the left and right contexts of a word, resulting in significantly better contextual representations.

My own experience with bert was largely around using it for document classification and more complex sequence labeling tasks. The pre-trained models offered by google were a game changer; it drastically reduced the amount of labeled training data required for good performance. The general approach involves fine-tuning the pre-trained bert model on our target task. For instance, you may want to use a sentence classification head for sentiment analysis.

Here’s a simplified example of using the `transformers` library from hugging face for text classification:

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Sample text data
texts = ["this movie is fantastic!", "a terrible experience", "i enjoyed it a lot", "awful acting"]
labels = [1, 0, 1, 0] # 1 for positive, 0 for negative

# Prepare data for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

# Convert the tensors to numpy array
input_ids = encoded_inputs['input_ids'].numpy()
attention_mask = encoded_inputs['attention_mask'].numpy()

# Split into training and testing
train_input_ids, test_input_ids, train_attention_mask, test_attention_mask, train_labels, test_labels = train_test_split(input_ids, attention_mask, np.array(labels), test_size=0.2)

# Model definition
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Training
model.fit(
    [train_input_ids, train_attention_mask], train_labels, epochs=2, batch_size=2, verbose=0
)

# Evaluation
eval_result = model.evaluate([test_input_ids, test_attention_mask], test_labels)
print(f"evaluation loss and acc {eval_result}")

# prediction
test_text = ["the acting was great"]
encoded_test_inputs = tokenizer(test_text, padding=True, truncation=True, return_tensors='tf')
prediction = model.predict([encoded_test_inputs['input_ids'], encoded_test_inputs['attention_mask']])
predicted_class = tf.argmax(prediction.logits, axis=1).numpy()
print(f"bert prediction {predicted_class}")

```

This code demonstrates the basic workflow of loading bert tokenizer, encoding the data into an input understandable by bert, and then using bert model on our data, finetuning it based on our input data. Note, that the main benefit of using bert is that we are utilizing the powerful features already trained by it, and then finetuning it based on our own data.

The critical difference between the cnn and bert approaches here lies in how the models process context. cnn essentially learns local dependencies while bert is able to process longer sequences and better understand global context due to the transformer attention mechanism. For basic tasks, like simple sentiment analysis, a fine-tuned cnn might be sufficient; however, for complex tasks or where longer contexts matter, bert offers significant advantages at the cost of higher computational complexity.

Finally, here's a quick snippet that shows how one could use bert to do feature generation, which can then be used downstream for clustering or other tasks:

```python
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np

# Sample text data
texts = ["this is a good movie", "bad review", "i love this", "not very good"]

# Prepare data for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

# Bert feature extraction
model = TFBertModel.from_pretrained('bert-base-uncased')
outputs = model(encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'])

# Get the hidden states of the last layer from bert model, use the CLS token for sequence level representation
features = outputs.last_hidden_state[:, 0, :]

print(f"bert feature extraction output {features}")
```

This final example highlights the versatility of bert. We’re using the bert model to extract contextual embeddings (features) for sentences, which can then be further utilized for downstream tasks like clustering or classification with other models. Here, we extract the CLS token's last hidden state output which is a good way to represent the whole sequence as one dense vector.

For those interested in deepening their understanding, i would strongly recommend the original bert paper ["bert: pre-training of deep bidirectional transformers for language understanding"] and ["attention is all you need"] which introduces transformer architecture. Furthermore, any comprehensive book on deep learning, such as "deep learning with python" by francois chollet, will also prove invaluable. "speech and language processing" by dan jurafsky and james martin is also an excellent reference book for foundations in nlp. Also, check the hugging face transformers documentation which is an incredible resource for understanding and utilizing transformer based models.

In conclusion, both cnn and bert offer powerful tools for text data processing; the specific selection depends largely on the complexities of the specific task at hand and available computational resources. While cnn provides a good and less compute-intensive approach to extract local patterns, bert, with its transformer backbone, is invaluable for tasks that require understanding the context at scale.
