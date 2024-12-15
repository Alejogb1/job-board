---
title: "How to do a Hierarchical LSTM autoencoder - model not training?"
date: "2024-12-15"
id: "how-to-do-a-hierarchical-lstm-autoencoder---model-not-training"
---

so, hierarchical lstm autoencoders, and they're not playing ball, training-wise. i've been there, trust me. it's a tricky area, and the "hierarchical" bit adds a layer of fun, doesn’t it? (that's my attempt at tech humor, i know, it's not gonna win any comedy awards). anyway, let's unpack this.

first, when you say "not training", what exactly does that mean? is the loss just not decreasing? is it fluctuating wildly? is it stuck at some high value? these details matter. a loss that's not moving is very different from a loss that's jumping around. it helps to be as precise as possible when diagnosing, and it is essential to see your loss curve to figure out the problem.

i've spent more nights than i care to remember staring at loss curves that looked like abstract art rather than actual training progress. the first time i tried a hierarchical lstm, i thought i’d cracked the code. i had a dataset of time series data, multiple variables, each with its own temporal dynamic. i designed a two-level lstm architecture, thinking this is genius. level one was supposed to capture lower-level time relationships within each variable, level two was supposed to capture the higher-level temporal relationships between those encoded variables. the loss? it was doing its own thing, not what i asked of it. turns out my data prep was… not ideal. so, we'll go through the usual suspects to troubleshoot this.

1. **data preprocessing matters big time:** hierarchical models, especially lstms, are notoriously sensitive to the data they're fed. if you have multiple time series, scaling them appropriately is vital. you can try standard scaler or min-max scaler. make sure the scales are consistent across all the time series. also, look at your time series lengths, padding sequences if they vary to a consistent length and ensuring no very short or missing sequences. it could be an issue where the sequences have an extremely large or extremely small variation of length. remember these lstm don’t handle well big variation on sequence length. think of it like trying to fit pieces of puzzle but some pieces are different sizes. it just won’t work. that was precisely my first problem with my data, some sequences were only 2 timesteps, while others were 100, the lstm had no idea what to do with that.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(sequences):
  """scales and pads variable length sequences to a max length"""
  max_len = max(len(seq) for seq in sequences)
  padded_sequences = []
  for seq in sequences:
    pad_len = max_len - len(seq)
    padded_seq = np.pad(seq, ((0, pad_len), (0, 0)), 'constant')
    padded_sequences.append(padded_seq)

  # scale each feature
  scaled_seqs = []
  for feature_index in range(padded_sequences[0].shape[1]):
      feature_values = [seq[:, feature_index] for seq in padded_sequences]
      all_values_feature= np.concatenate(feature_values)
      scaler = StandardScaler()
      all_values_feature_scaled = scaler.fit_transform(all_values_feature.reshape(-1,1))
      start_index=0
      for seq_index in range(len(padded_sequences)):
        end_index=start_index + len(padded_sequences[seq_index])
        padded_sequences[seq_index][:,feature_index]= all_values_feature_scaled[start_index:end_index].flatten()
        start_index=end_index
  return np.array(padded_sequences)

# example usage:
# assuming sequences is a list of numpy arrays, where each array
# is a time series with dimensions (time_steps, features)
# example data for demonstration:
sequences = [
    np.random.rand(50, 3),
    np.random.rand(70, 3),
    np.random.rand(20, 3)
]
processed_sequences = preprocess_data(sequences)
print(processed_sequences.shape) #will output (3, 70, 3) on this example
```

2. **architecture check:** let's talk about your layers. are they too deep? too shallow? lstms, especially when stacked, have a tendency to be finicky. and don't forget the input dimension. in my case, i was feeding the second lstm layer with the encoded outputs of the first lstm, but the dimensions were not matching. i made the output of the first layer the correct dimension, using a time-distributed layer on top of the first lstm output. also, the size of the hidden state matters, too large and the network will overfit, too small it won’t be able to learn anything. i usually start with the hidden dimension of the first lstm to be the same size of the number of features of your dataset.

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_hierarchical_lstm_autoencoder(input_shape, latent_dim, first_lstm_dim):
    """builds a hierarchical lstm autoencoder model."""
    inputs = layers.Input(shape=input_shape)

    # encoder layer 1: lstm for individual time series
    enc_lstm_1 = layers.LSTM(first_lstm_dim, return_sequences=True)(inputs)
    enc_lstm_1_output = layers.TimeDistributed(layers.Dense(latent_dim))(enc_lstm_1)

    # encoder layer 2: lstm for encoded sequences
    enc_lstm_2 = layers.LSTM(latent_dim, return_state=True)
    enc_output, state_h, state_c = enc_lstm_2(enc_lstm_1_output)
    encoder_states = [state_h, state_c] # store states for decoder initialization

    # decoder layer 1: lstm
    dec_lstm_1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_outputs, _, _ = dec_lstm_1(layers.RepeatVector(input_shape[0])(enc_output), initial_state=encoder_states)
    dec_lstm_1_output = layers.TimeDistributed(layers.Dense(first_lstm_dim))(dec_outputs)

    # decoder layer 2: lstm
    dec_lstm_2 = layers.LSTM(first_lstm_dim, return_sequences=True)
    decoded = dec_lstm_2(dec_lstm_1_output)

    # time distributed output
    outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(decoded)

    model = tf.keras.Model(inputs, outputs)
    return model

# example
input_shape = (70,3) # max sequence length and num features from example above
latent_dim = 10
first_lstm_dim= 3
model = build_hierarchical_lstm_autoencoder(input_shape, latent_dim, first_lstm_dim)
model.summary()
```

3. **optimizer and learning rate**: this is where a lot of problems arise. are you using a standard optimizer like adam? that’s usually the first bet. but the learning rate is a parameter that needs to be carefully tuned. a learning rate too high and the optimization process can diverge and never converge, too low and the training process will take forever to see any kind of movement. try learning rates between 0.001 to 0.00001. and also check the batch size. if you are training in sequences, the batch size should not be too large because the gradient might not be as informative.

```python
# compiling the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse') # mse is good for time series data
# training the model
processed_sequences= preprocess_data(sequences)
model.fit(processed_sequences,processed_sequences, epochs=100, batch_size=16) #epochs will depend on data complexity

```

4. **loss function:** mse is fine for many cases when you are doing time-series data reconstruction. but you could also try mae. you might also consider cosine loss in some situations. mse usually converges quicker but mae is more robust to outliers. the important here is to see if your loss function is appropriate to the problem.

5. **regularization:** lstms are prone to overfitting and in this case the network would not generalize and reconstruct the input properly. using a dropout layer or an l2 regularizer on your layers are a must in this cases. dropout would drop a proportion of neurons randomly and thus allow generalization. l2 regularization penalizes the weights to prevent large and small parameters that usually are the sign of an overfitted network.

6. **dataset size:** i've often thought of it that my models are hungry, they need to be fed to learn anything. if your dataset is too small the model won’t have enough information to generalize. it will either overfit or the gradients will be all over the place. sometimes, data augmentation is the solution. but first you must ensure your dataset is as large as possible. there are also things such as synthetic datasets, which are also useful, depending on the problem.

as for recommendations for more in-depth knowledge, i would point you towards deep learning textbooks, like "deep learning" by goodfellow, bengio, and courville, this is like the bible for deep learning. for specific lstm stuff and sequence modeling, try "sequence to sequence learning with neural networks" by sutskever, vinyals, and le. these are not easy reading material but you will get an idea of how these models work in the inside, giving you a better intuition when debugging your training process.

i hope this helps you. it's a process, and it's easy to get stuck. keep iterating, keep testing, and most importantly, keep an eye on that loss curve. good luck!
