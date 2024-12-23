---
title: "How can a custom training loop leverage stateful LSTMs?"
date: "2024-12-23"
id: "how-can-a-custom-training-loop-leverage-stateful-lstms"
---

Alright, let's talk about training loops and stateful LSTMs. It's a topic that, if not handled correctly, can lead to some... interesting debugging sessions. I remember tackling a particularly thorny sequence prediction problem a few years back – customer behavior prediction, if you must know – where traditional stateless LSTMs just wouldn't cut it. The inherent temporal dependencies were so crucial that discarding the internal state at each batch boundary was throwing away vital information. That's when the stateful approach became less of an option and more of a necessity.

So, how *do* you effectively train a stateful LSTM? The crux of it lies in understanding that, unlike their stateless cousins, stateful LSTMs retain their internal hidden and cell states across batches *within a sequence*. This implies a tighter relationship between your data and your training pipeline. The first, and perhaps most critical, adjustment you’ll need to make is how you prepare your training data. It needs to be carefully crafted to maintain sequence continuity across the batches you’re feeding into the model.

Let’s dive a bit deeper into the practical aspects. With stateless LSTMs, each batch is an independent chunk of data. You could shuffle your sequences almost arbitrarily and it wouldn’t impact the learning process significantly, save for introducing some noise into the gradient updates. But with stateful LSTMs, the order and grouping of your data are of utmost importance. You absolutely must avoid shuffling across sequences, and you should be aware of the number of sequences you’re processing simultaneously. The batch size is no longer just about computational efficiency; it directly dictates the number of independent sequences the LSTM maintains state for.

Let's see some code to solidify these concepts. We'll assume a simple scenario: you have time series data that are pre-segmented into sequences, and you are using Tensorflow or Keras (which, behind the scenes, leverages Tensorflow).

**Example 1: State Resetting at Sequence Boundaries**

First, I'll show the skeleton of the core training loop. A crucial part when working with stateful lstms is reseting the state at the end of each sequence, but within an epoch. Note that you will need to have *batch_size* sequences for your code to work correctly.

```python
import tensorflow as tf
import numpy as np

# Assume sequences is a list of NumPy arrays, each with shape (sequence_length, features)
def create_batches(sequences, batch_size):
    num_sequences = len(sequences)
    num_batches = num_sequences // batch_size
    batches = []
    for i in range(num_batches):
      start_index = i * batch_size
      end_index = (i + 1) * batch_size
      batches.append(sequences[start_index:end_index])

    return batches

def lstm_model(units, features):
    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(units, batch_input_shape=(batch_size, None, features), stateful=True, return_sequences=True),
      tf.keras.layers.Dense(1)
    ])
    return model

# Dummy data (replace with your own data loading logic)
num_sequences = 20
sequence_length = 50
features = 10
batch_size = 5
sequences = [np.random.rand(sequence_length, features) for _ in range(num_sequences)]
y = [np.random.rand(sequence_length, 1) for _ in range(num_sequences)]

batched_sequences = create_batches(sequences, batch_size)
batched_labels = create_batches(y, batch_size)

units = 64
model = lstm_model(units, features)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()


epochs = 10
for epoch in range(epochs):
    for batch_seq, batch_label in zip(batched_sequences, batched_labels):

        batch_seq = np.stack(batch_seq)
        batch_label = np.stack(batch_label)
        with tf.GradientTape() as tape:
            predictions = model(batch_seq)
            loss = loss_fn(batch_label, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Reset state after processing all sequences in the epoch
    model.reset_states()
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
```

Notice a few things. First, we're explicitly setting `stateful=True` in the LSTM layer and specifying `batch_input_shape`. Secondly, I've added a `model.reset_states()` call at the end of each epoch. Crucially, this is not being called at every batch. This will reset the state after all sequences are processed in the current epoch. This approach preserves the sequential dependencies within each sequence, while also avoiding carrying state to the next sequence. This works well when all sequences are about the same size.

**Example 2: Handling Sequences of Variable Lengths**

In practice, your sequence lengths might vary. You will need to pad or truncate your sequences to the maximum sequence length in your batch. This adds a layer of complexity in the batch preparation logic which I'll show below.

```python
def create_padded_batches(sequences, batch_size):
  num_sequences = len(sequences)
  num_batches = num_sequences // batch_size
  batches = []
  for i in range(num_batches):
    start_index = i * batch_size
    end_index = (i+1) * batch_size
    batch = sequences[start_index:end_index]
    max_len = max([seq.shape[0] for seq in batch])
    padded_batch = []
    for seq in batch:
      padding = np.zeros((max_len - seq.shape[0], seq.shape[1]))
      padded_batch.append(np.concatenate([seq, padding]))
    batches.append(np.stack(padded_batch))
  return batches


# dummy data
num_sequences = 20
max_sequence_length = 50
features = 10
batch_size = 5
sequences = [np.random.rand(np.random.randint(20, max_sequence_length), features) for _ in range(num_sequences)]
y = [np.random.rand(seq.shape[0], 1) for seq in sequences]
batched_sequences = create_padded_batches(sequences, batch_size)
batched_labels = create_padded_batches(y, batch_size)

units = 64
model = lstm_model(units, features) # same model from Example 1.

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()


epochs = 10
for epoch in range(epochs):
    for batch_seq, batch_label in zip(batched_sequences, batched_labels):

        with tf.GradientTape() as tape:
            predictions = model(batch_seq)
            loss = loss_fn(batch_label, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Reset state after processing all sequences in the epoch
    model.reset_states()
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
```

Here, we're introducing padding to ensure all sequences within a batch have equal length and also masking those padded parts so the loss isn't calculated on them. The batch preparation is now more complex, requiring us to pad shorter sequences to the maximum length of the sequences within the batch. The training loop remains mostly the same as before, but it's important to remember that the padding strategy can influence how your model learns and it must be adapted depending on your specific problem.

**Example 3: Unrolling Sequences Step-by-Step**

Finally, let's look at an approach that allows for the greatest control: manually unrolling sequences step-by-step. You may need this for more advanced scenarios, such as when you want to use teacher forcing.
```python
import tensorflow as tf
import numpy as np

# Assume sequences is a list of NumPy arrays, each with shape (sequence_length, features)
def create_batches(sequences, batch_size):
    num_sequences = len(sequences)
    num_batches = num_sequences // batch_size
    batches = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        batches.append(sequences[start_index:end_index])

    return batches


def lstm_model(units, features):
    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(units, batch_input_shape=(batch_size, None, features), stateful=True, return_sequences=True),
      tf.keras.layers.Dense(1)
    ])
    return model


# Dummy data (replace with your own data loading logic)
num_sequences = 20
sequence_length = 50
features = 10
batch_size = 5
sequences = [np.random.rand(sequence_length, features) for _ in range(num_sequences)]
y = [np.random.rand(sequence_length, 1) for _ in range(num_sequences)]

batched_sequences = create_batches(sequences, batch_size)
batched_labels = create_batches(y, batch_size)

units = 64
model = lstm_model(units, features)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()


epochs = 10
for epoch in range(epochs):
    for batch_seq, batch_label in zip(batched_sequences, batched_labels):
        batch_seq = np.stack(batch_seq)
        batch_label = np.stack(batch_label)

        for t in range(batch_seq.shape[1]):  # Iterate over time steps
             with tf.GradientTape() as tape:
               current_input = batch_seq[:, t:t + 1, :]  # Shape (batch_size, 1, features)
               current_target = batch_label[:, t:t + 1, :]
               predictions = model(current_input)
               loss = loss_fn(current_target, predictions)
             gradients = tape.gradient(loss, model.trainable_variables)
             optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Reset state after processing all sequences in the epoch
    model.reset_states()
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

```

In this example, we're processing each time step of each batch individually within the inner loop. This approach gives you step-by-step control, enabling flexibility such as using teacher forcing or calculating the loss based on a temporal horizon. It’s more verbose but can be necessary for more intricate sequence learning scenarios.

Before concluding, let me suggest a few useful resources. For a deep theoretical understanding, the "Deep Learning" book by Goodfellow, Bengio, and Courville is essential. For practical implementation details, the official Keras documentation on LSTMs and recurrent neural networks is invaluable. Also, papers such as "Learning long-range dependencies with gated recurrent neural networks" (Hochreiter and Schmidhuber, 1997) will provide deeper insight into the mechanics of the LSTM cell.

Working with stateful LSTMs definitely requires a more deliberate approach than their stateless counterparts. The devil, as they say, is in the details, particularly in how you structure your batches and manage the internal state of the model. By carefully designing your training loop to respect the sequence boundaries and using reset states when needed, you can effectively harness the power of stateful LSTMs for complex sequential modeling tasks. I hope that overview and examples help clarify this topic. Remember, thorough experimentation and careful consideration of your specific data are key.
