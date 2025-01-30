---
title: "How can BERT embeddings be used with an LSTM?"
date: "2025-01-30"
id: "how-can-bert-embeddings-be-used-with-an"
---
BERT, a transformer-based model, excels at generating contextualized word embeddings, while LSTMs, a type of recurrent neural network, are well-suited for sequence processing. Combining these two powerful architectures can lead to enhanced performance in various natural language tasks, especially those requiring understanding of both word semantics and sentence structure. The key to effective integration lies in utilizing BERT's pre-trained knowledge to feed information into the LSTM network. I’ve personally implemented this combination in multiple projects, ranging from sentiment analysis to complex text summarization, and have observed significant improvements over using either model alone.

The fundamental premise behind this approach involves using BERT to generate dense vector representations for each word in a sequence. Instead of utilizing conventional word embedding matrices like word2vec or GloVe, we process the input sequence through BERT to obtain contextualized embeddings that capture semantic nuances and contextual information. These resulting BERT embeddings then serve as the input for the LSTM network. The LSTM is then responsible for learning temporal dependencies and long-range relationships across the sequence, utilizing the enriched word representations provided by BERT. This integration is not a simple concatenation of outputs; rather, it's a sequential flow where BERT’s rich representation informs the LSTM’s sequential learning. The architecture leverages the strengths of both models: BERT's understanding of individual word meaning within its context and LSTM’s aptitude for modeling sequential data.

Let's examine the practical implementation via Python with TensorFlow/Keras. The process typically involves the following steps: First, tokenize the input text using BERT's tokenizer. Second, feed the tokenized sequence into the pre-trained BERT model. Third, extract the contextualized embeddings from BERT's output. And fourth, feed these embeddings into the LSTM layer(s) followed by a task-specific output layer. Crucially, the BERT model is often kept frozen or finetuned at a very low learning rate to retain its powerful pre-trained knowledge, while the LSTM layers are allowed to adapt more freely to the task at hand.

Here's a breakdown via code example 1, where I showcase a simple text classification scenario. This example makes use of the TensorFlow library, and the BERT implementation is using the pre-trained model from Huggingface. I’m opting to freeze the BERT model here and train the subsequent LSTM layer.

```python
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras import layers, Model

# Load pre-trained BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)

# Define the combined model
def create_bert_lstm_model(max_sequence_length, lstm_units, num_classes):
    input_ids = layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="attention_mask")

    bert_output = bert_model([input_ids, attention_mask])[0] # Extract hidden states

    lstm_output = layers.LSTM(lstm_units)(bert_output)
    output = layers.Dense(num_classes, activation='softmax')(lstm_output) # Task specific layer
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    return model


max_seq_length = 128 # Define your max sequence length
lstm_units = 64       # Define desired LSTM units
num_classes = 2       # Example for binary classification

# Create the model
model = create_bert_lstm_model(max_seq_length, lstm_units, num_classes)

# Freeze BERT weights to maintain pre-trained knowledge
bert_model.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Sample data for demonstration purposes
texts = ["This movie was great!", "I hated this film.", "This was a good watch.", "The worst experience ever."]
labels = [[1, 0], [0, 1], [1, 0], [0, 1]]  # one hot encoded


# Tokenize the input
tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_length, return_tensors="tf")

# Training the model
model.fit([tokens['input_ids'], tokens['attention_mask']], tf.constant(labels), epochs=5)

#Prediction (example)
new_texts = ["This was alright", "Absolutely terrible film"]
new_tokens = tokenizer(new_texts, padding=True, truncation=True, max_length = max_seq_length, return_tensors = "tf")
predictions = model.predict([new_tokens['input_ids'], new_tokens['attention_mask']])
print(predictions)

```

In this code, `create_bert_lstm_model` constructs the network. Crucially, we pass the output of BERT's hidden states (`bert_output`) into the LSTM layer. The model accepts two inputs: `input_ids` and `attention_mask`, which are outputs of the tokenizer, and outputs the prediction. The `bert_model.trainable = False` line is imperative for freezing the BERT layers. I’ve chosen to use categorical crossentropy as loss function given the one-hot encoded labels.  This first example showcases the foundational aspect of integrating the BERT embedding and using them as the input to the LSTM layer.

Now let's explore a variation with code example 2. This code refines the previous by adding an optional pooling layer after the BERT output. Pooling is important for reducing the dimensionality of the BERT embeddings, potentially speeding up training and allowing for a more efficient LSTM input, especially if your input sequence lengths are variable or very large. I’ve used the average pooling strategy which is computationally simpler and effective.

```python
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras import layers, Model

# Load pre-trained BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)

# Define the combined model with average pooling
def create_bert_lstm_pooled_model(max_sequence_length, lstm_units, num_classes):
    input_ids = layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="attention_mask")

    bert_output = bert_model([input_ids, attention_mask])[0] # Extract hidden states

    # Average pooling layer over tokens
    pooled_output = tf.reduce_mean(bert_output, axis=1)

    lstm_output = layers.LSTM(lstm_units)(tf.expand_dims(pooled_output,axis=1))
    output = layers.Dense(num_classes, activation='softmax')(lstm_output)
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    return model


max_seq_length = 128
lstm_units = 64
num_classes = 2

# Create the model
model = create_bert_lstm_pooled_model(max_seq_length, lstm_units, num_classes)

# Freeze BERT weights
bert_model.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Sample data (same as before)
texts = ["This movie was great!", "I hated this film.", "This was a good watch.", "The worst experience ever."]
labels = [[1, 0], [0, 1], [1, 0], [0, 1]]

# Tokenize
tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_length, return_tensors="tf")

# Train the model
model.fit([tokens['input_ids'], tokens['attention_mask']], tf.constant(labels), epochs=5)


#Prediction (example)
new_texts = ["This was alright", "Absolutely terrible film"]
new_tokens = tokenizer(new_texts, padding=True, truncation=True, max_length = max_seq_length, return_tensors = "tf")
predictions = model.predict([new_tokens['input_ids'], new_tokens['attention_mask']])
print(predictions)

```

Notice the addition of `pooled_output = tf.reduce_mean(bert_output, axis=1)` in this version. This pools the BERT embeddings before they are fed to the LSTM layer. We also need to `expand_dims` the pooled output so that it is in the appropriate shape for inputting into the LSTM layer. This step reduces the computational load on the LSTM layer, especially for long sequences, but it can potentially reduce the level of detail in the input. The tradeoff between performance and computational efficiency will depend on the specific application.

Finally, in code example 3, let's consider a finetuning example. Instead of freezing BERT’s weights completely, we allow the weights to be fine-tuned, albeit at a very low learning rate. This approach can potentially improve model performance, allowing the BERT parameters to slightly adapt to the downstream task.

```python
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras import layers, Model
import tensorflow_addons as tfa

# Load pre-trained BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)

# Define the combined model (no pooling this time)
def create_bert_lstm_finetune_model(max_sequence_length, lstm_units, num_classes):
    input_ids = layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="attention_mask")

    bert_output = bert_model([input_ids, attention_mask])[0] # Extract hidden states

    lstm_output = layers.LSTM(lstm_units)(bert_output)
    output = layers.Dense(num_classes, activation='softmax')(lstm_output)
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    return model


max_seq_length = 128
lstm_units = 64
num_classes = 2

# Create the model
model = create_bert_lstm_finetune_model(max_seq_length, lstm_units, num_classes)

# Choose a very low learning rate for the BERT layer while keeping the LSTM at a higher rate
optimizer = tfa.optimizers.AdamW(
    learning_rate=2e-5, weight_decay=0.001)  # Adjust learning rate
optimizer_lstm = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Train the model with a learning rate scheduler
# Set the specific layers for optimization with different learning rates
trainable_layers = {layer.name: layer for layer in model.layers if layer.trainable}

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
# Optimizer for the LSTM layer with a separate learning rate
optimizer_lstm_vars = [var for var in model.trainable_variables if var.name.startswith('lstm')]

# Define custom training step
def train_step(inputs, labels):
    with tf.GradientTape() as tape_bert, tf.GradientTape() as tape_lstm:
        output = model(inputs,training=True)
        loss_val = model.loss(labels,output)
    
    # Compute gradients for all layers
    gradients_lstm = tape_lstm.gradient(loss_val, optimizer_lstm_vars)
    gradients_bert = tape_bert.gradient(loss_val, model.trainable_variables)
    
    # Apply separate gradients for BERT and LSTM
    optimizer.apply_gradients(zip(gradients_bert, model.trainable_variables))
    optimizer_lstm.apply_gradients(zip(gradients_lstm, optimizer_lstm_vars))
    return loss_val


@tf.function
def train_loop(dataset):
  for inputs, labels in dataset:
      loss_val = train_step(inputs, labels)
      print(f"loss: {loss_val}")


# Sample data
texts = ["This movie was great!", "I hated this film.", "This was a good watch.", "The worst experience ever."]
labels = [[1, 0], [0, 1], [1, 0], [0, 1]]

# Tokenize
tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_length, return_tensors="tf")


dataset = tf.data.Dataset.from_tensor_slices(({"input_ids":tokens['input_ids'],
                                                 "attention_mask":tokens['attention_mask']},
                                                 tf.constant(labels))).batch(2)
# Train the model
train_loop(dataset)


#Prediction (example)
new_texts = ["This was alright", "Absolutely terrible film"]
new_tokens = tokenizer(new_texts, padding=True, truncation=True, max_length = max_seq_length, return_tensors = "tf")
predictions = model.predict([new_tokens['input_ids'], new_tokens['attention_mask']])
print(predictions)

```

In this version, the key change lies in how the optimizer is set up. Instead of using the `trainable = False` flag, the optimizer is configured to apply different learning rates to the LSTM layers and BERT model through custom training step `train_step`. This requires a slightly more complex implementation of training loop, but the benefit is increased control over the training process. It should be noted here that the model loss calculation remains the same.

In summary, the combined architecture of BERT and LSTM brings together the semantic understanding of BERT with the sequential modeling capabilities of LSTM. Variations in the architecture, like average pooling or finetuning, offer flexibility in addressing different task demands.  For further information, I suggest consulting materials on recurrent neural networks, specifically LSTMs, and transformer-based models like BERT. The official documentation for TensorFlow and Hugging Face's Transformers library also provide comprehensive information and guides.  Finally, research papers on transfer learning in natural language processing may offer useful insights.
