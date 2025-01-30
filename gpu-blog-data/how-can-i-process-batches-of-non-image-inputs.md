---
title: "How can I process batches of non-image inputs with varying shapes without padding?"
date: "2025-01-30"
id: "how-can-i-process-batches-of-non-image-inputs"
---
Batch processing of variable-shaped, non-image data without padding introduces complexities that traditional batching strategies often fail to address effectively. A core challenge lies in the expectation of fixed-shape tensors during parallel computation, which is inherent in frameworks like TensorFlow and PyTorch. Padding, while a common solution, can introduce bias or inefficiency, especially when the padded portion carries no meaningful information. To process such data effectively, I’ve found that utilizing a combination of dynamic batch construction and masking or sequence packing strategies provides optimal results.

The fundamental issue arises because neural network layers, particularly those implemented in optimized numerical libraries, are designed to operate on tensors with static shapes. Batch processing enables efficient computation by vectorizing operations across multiple data samples. However, when each sample in the batch has a different shape, direct tensor stacking becomes impossible without either padding (filling with zeros to reach a uniform length) or reshaping to the lowest common denominator dimension. Padding is particularly wasteful if the data has a large variation in size. To address this, one must abandon a naive stacking approach in favor of custom handling during the data loading and forward pass steps.

A typical scenario illustrating this might be processing sequences of events, where each event sequence may have a variable number of events. Suppose we're trying to use a recurrent neural network (RNN), such as an LSTM or GRU, to analyze these sequences. Direct stacking without padding would lead to shape mismatch errors. The solution involves constructing batches where samples with similar sequence lengths are grouped together. In practice, this is achieved by initially sorting the input data by their sequence lengths. Then, within each batch, the samples are stacked. Because the samples within a batch are of similar length, the amount of padding needed after stack is kept very low.

Instead of padding all sequences to the maximum length, we use sequence packing (or masking) to communicate to the neural network which time steps contain valid data within a sequence. Sequence packing, in the case of RNNs, can often be achieved using utilities provided by frameworks, for instance `torch.nn.utils.rnn.pack_padded_sequence` in PyTorch. Masking, in other context such as transformer encoder processing, involves generating an attention mask that is passed alongside input tensor and ensures that the transformer ignore padded elements. The choice depends on the type of model being used. In contrast, padding without using masking can lead the model to learn patterns or pay attention to padded information, which is undesirable.

To illustrate these concepts, let's examine three different code examples using Python and popular machine learning frameworks: one for preprocessing, one for RNN processing with sequence packing, and one for transformer encoder masking.

**Example 1: Dynamic Batch Construction (Preprocessing)**

```python
import numpy as np

def prepare_batches(data, batch_size):
    """
    Sorts input data by length and creates batches with similar lengths.

    Args:
        data: A list of sequences, where each sequence is a list of numbers.
        batch_size: The desired batch size.

    Returns:
        A list of lists representing batches of indices.
    """
    indexed_data = list(enumerate(data))
    sorted_data = sorted(indexed_data, key=lambda x: len(x[1]))
    indices = [idx for idx, _ in sorted_data]
    batches = []
    for i in range(0, len(indices), batch_size):
        batches.append(indices[i:i + batch_size])
    return batches

# Example usage
data = [
    [1, 2, 3],
    [4, 5, 6, 7, 8],
    [9, 10],
    [11, 12, 13, 14],
    [15],
    [16,17,18,19,20,21],
    [22,23,24,25]
]
batch_size = 2
batches = prepare_batches(data, batch_size)
print(f"Batches (indices): {batches}")

# Expected Output (varies slightly depending on the sorting algorithm stability)
# Batches (indices): [[4, 2], [0, 3], [6, 1], [5]]
```
This code constructs the batches based on similar sequence length. The sorting step, coupled with the batch creation logic, ensures that within each batch the variability in sequence length is minimized.  This reduces any wasted memory from padding within the batch. Note that padding is *still* applied on the data within a batch, but it is now minimized. Also note that the return value of `prepare_batches()` is not the batches themselves, but the list of indices that we will use later to access the original data during training. This separation is helpful to decouple the batch construction logic from the data loading part.

**Example 2: RNN Processing with Sequence Packing (PyTorch)**

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

class SequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SequenceModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequences, lengths):
        embedded = self.embedding(sequences)
        packed = pack_sequence(embedded, lengths.cpu(), enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        output = self.fc(hidden[-1])
        return output

# Example Usage
input_dim = 26  # Assuming 26 possible inputs
hidden_dim = 32
output_dim = 2
model = SequenceModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Assuming batches are generated using the previous example
batches = [[4, 2], [0, 3], [6, 1], [5]]
original_data = data # Using the same variable from the previous example

# Example Training Loop
for batch_indices in batches:
    batch_sequences = [torch.tensor(original_data[i], dtype=torch.long) for i in batch_indices]
    lengths = torch.tensor([len(seq) for seq in batch_sequences], dtype=torch.long)
    padded_sequences = nn.utils.rnn.pad_sequence(batch_sequences, batch_first=True)
    labels = torch.randint(0, output_dim, (len(batch_sequences),)) # Dummy labels
    
    optimizer.zero_grad()
    output = model(padded_sequences, lengths)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    print(f"Batch loss: {loss.item():.4f}")
```

This PyTorch example demonstrates sequence packing, which efficiently handles variable sequence lengths within a batch. The `pack_sequence` function combines the sequences into a packed representation, removing padding from the computation during the LSTM operation. It requires `lengths` to indicate sequence lengths for every batch member. Note that even though `pack_sequence` avoids using zero values in the padded sequence, we still need to do padding of the batch using `pad_sequence`. But, the padding is minimal due to the batch generation logic shown in example one. The output of the LSTM is handled appropriately to use only the output vector of the last time step from each sequence.

**Example 3: Transformer Encoder with Masking (TensorFlow)**
```python
import tensorflow as tf

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size):
        super(TransformerEncoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
    def call(self, inputs, mask):
        x = self.embedding(inputs)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x
        
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask):
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Example Usage
num_layers = 2
d_model = 32
num_heads = 2
dff = 64
vocab_size = 26
encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, vocab_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Assuming batches are generated using the first example.
batches = [[4, 2], [0, 3], [6, 1], [5]]
original_data = data # Using the same variable from the previous example

# Example Training Loop
for batch_indices in batches:
    batch_sequences = [tf.convert_to_tensor(original_data[i], dtype=tf.int32) for i in batch_indices]
    lengths = [len(seq) for seq in batch_sequences]
    padded_sequences = tf.keras.utils.pad_sequences(batch_sequences, padding='post')
    mask = tf.sequence_mask(lengths, maxlen=padded_sequences.shape[1])
    labels = tf.random.uniform(shape=(len(batch_sequences),), minval=0, maxval=vocab_size, dtype=tf.int32) # Dummy labels

    with tf.GradientTape() as tape:
        output = encoder(padded_sequences, mask)
        loss = loss_fn(labels, output[:,0,:])
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

    print(f"Batch loss: {loss.numpy():.4f}")
```
This TensorFlow example shows how to use a masking technique within a transformer encoder. We generate a mask by checking which values are valid and which have been added through post padding. The mask is passed to the transformer layers, which in turn ignores the padded values when computing the attention scores. This avoids the model from learning from padded values.

For further study, I would suggest exploring research on variable sequence length processing, particularly as it pertains to RNNs and Transformers. Familiarize yourself with the documentation on sequence packing functions in the major deep learning frameworks and learn about attention mechanism masking, both are crucial to understanding how to deal with non-padded batches. Furthermore, a theoretical background of deep learning frameworks’ underlying design principle can help understanding the need for tensor with fixed shape during batch processing. Finally, investigate libraries that provide utilities for dynamic batching, potentially including more sophisticated bucket-based sorting or even reinforcement learning-based batching. This will further enhance your understanding and flexibility when dealing with diverse data processing requirements.
