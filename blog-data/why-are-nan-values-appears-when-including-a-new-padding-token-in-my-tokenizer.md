---
title: "Why are NAN values appears when including a new padding token in my tokenizer?"
date: "2024-12-15"
id: "why-are-nan-values-appears-when-including-a-new-padding-token-in-my-tokenizer"
---

alright, so you’re seeing nan values pop up after adding a new padding token to your tokenizer, huh? i’ve been there, done that, got the t-shirt and probably spilled coffee on it too. it's a classic gotcha when you're messing with vocabulary and embeddings, especially in natural language processing stuff. let me break down what's likely going on and how i’ve typically tackled it in past projects.

the core problem usually boils down to how your model is handling the initial embedding vectors for the newly introduced token, specifically the padding token. when you add a new token to your tokenizer's vocabulary, a few things happen under the hood. first the tokenizer creates a new mapping between this new string or token and an integer index, lets say, ‘0’ for padding, now this integer will be used to look up the vector representation of this token. now, your model needs some way of assigning vector values for that new entry, this typically happen in two ways: one by initializing it randomly, or second, using the pre trained values.

if you're not careful, this initialization step can lead to nan values. why? well, consider the scenario, you have a pre-trained model which was trained with a given vocabulary size and embeddings. now, you have a tokenizer mapping the strings, or words to integer indices, these indices are what your embedding layers uses to get the vector for that token. so, when you add new tokens, those tokens do not have a pre-trained vectors, because the pre-trained model never saw these. what typically happens is that this new vector, is initialized randomly, if the embedding space was pre-trained by another tokenizer, your pre-trained embedding will not have this new token, so the embedding layer may use default values for that token in the lookup table. the most common default value is zero initialization, which causes problems during learning, specially if you are doing calculations on it using floating points.

most commonly the nan appears during training in your loss function calculation or backpropagation step because you are using an optimization function like stochastic gradient descent with a learning rate where the gradient for that new token’s embedding is large. and since the vector was initialized randomly, or zeroed, it causes the values to explode and become nan. a large enough gradient, even on a vector component that is close to zero, can easily push the value into the nan territory with float arithmetic. it's like trying to divide by an infinitesimally small number – it goes boom.

i've encountered this in a few cases. one time, i was working on a sentiment analysis model using a transformer architecture, adding specific padding tokens for batch processing of variable length sequences. i thought i was being clever, just adding the new token and not modifying the embedding layer at all, but boom, nan values. that's when i learned the hard way about vector initialization and gradient behavior. i spent a good part of my weekend debugging that. i had to actually check each layer and its values to actually pin down the exact layer that was causing the nan, you can say it was a fun weekend.

another time it happened to me was when i was fine tuning a text generation model for a very specific text domain. the original tokenizer did not have very domain specific words, i decided to add them to the vocabulary, and i made the assumption that the pre-trained embeddings would magically adjust, but that did not happen.

here's the breakdown with code examples on how to avoid it. let’s assume we’re using python with pytorch or tensorflow, since those are my daily drivers, i also use jax sometimes but lets keep it to python for this explanation.

first, let’s look at a scenario where you are using pytorch and you did not take care of the padding token initialization when creating the embedding:

```python
import torch
import torch.nn as nn

vocab_size = 100 # the original vocab size
embedding_dim = 128
pad_token_id = 0 # lets say you added a padding token at index 0 in your tokenizer

# incorrect initialization, new pad vector will be zeros
embedding_layer = nn.Embedding(vocab_size + 1, embedding_dim)

# lets pretend this is our sequence batch and they are padded
input_batch = torch.tensor([[1, 2, 3, pad_token_id], [4, 5, pad_token_id, pad_token_id]])

#forward pass with embedding
embedded = embedding_layer(input_batch)

print(embedded)
```

in this incorrect example, when a new padding token is added to the vocabulary, the embedding layer is initialized, and default zero initialization is used for the padding token. as explained above this may cause issues later during training, in addition in this case it was not even specified that the padding token would be used, thus during training it was not updated as part of the gradient update.

now, lets see how to properly initialize this in pytorch.

```python
import torch
import torch.nn as nn
import numpy as np

vocab_size = 100
embedding_dim = 128
pad_token_id = 0

# load existing embedding, lets pretend you loaded a file with pre trained vectors
pretrained_embeddings = np.random.rand(vocab_size, embedding_dim)

#correct initialization
embedding_layer = nn.Embedding(vocab_size + 1, embedding_dim)

# copy the pre-trained embeddings to the new layer.
embedding_layer.weight.data[:vocab_size] = torch.from_numpy(pretrained_embeddings)

# use something more sensible than zeros, like a random normal vector
new_vector = torch.randn(embedding_dim) # or torch.zeros(embedding_dim) if you want to zero initialize
embedding_layer.weight.data[pad_token_id] = new_vector

# use this to indicate your padding token in your loss function, example below
padding_idx = pad_token_id

# lets pretend this is our sequence batch and they are padded
input_batch = torch.tensor([[1, 2, 3, pad_token_id], [4, 5, pad_token_id, pad_token_id]])

#forward pass with embedding
embedded = embedding_layer(input_batch)

print(embedded)
```

in this updated example, a new embedding layer is created with the increased vocab size, then it loads the previous embeddings if there are any, and finally it initializes the new padding token with a random tensor, you could even zero initialize it, just make sure you are not trying to update it with the gradients by using the `padding_idx` in your loss function. the `padding_idx` parameter is used by pytorch to mask out the values in the padded positions.

here is the same concept using tensorflow.

```python
import tensorflow as tf
import numpy as np

vocab_size = 100
embedding_dim = 128
pad_token_id = 0

# load existing embedding, lets pretend you loaded a file with pre trained vectors
pretrained_embeddings = np.random.rand(vocab_size, embedding_dim)

#correct initialization
embedding_layer = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(
        np.concatenate([pretrained_embeddings, np.random.rand(1,embedding_dim)], axis=0)
    ),
    mask_zero = True # set this to mask padding
    )

# lets pretend this is our sequence batch and they are padded
input_batch = tf.constant([[1, 2, 3, pad_token_id], [4, 5, pad_token_id, pad_token_id]])

#forward pass with embedding
embedded = embedding_layer(input_batch)

print(embedded)
```

this tensorflow code does something similar, first creates the new layer with the increased vocabulary size, then initializes it using the previously trained embeddings and the new random one using the `Constant` initializer, and lastly uses `mask_zero = True` to mask the values of the padded tokens and also prevent their gradient update.

so to recap, you need to ensure that your newly added padding token has a reasonable initialization and the padding values are properly masked in the loss function or during training, and not just set to zero. i usually try a random normal initialization before using zero initialization. the mask parameter is usually the key part here. because you might end up updating this vector in the gradient step and it may explode into nans.

also, consider if your embedding space is pre-trained or not. if you loaded weights from a checkpoint, the embedding table may not be the size you expected because it does not account for the new tokens, so you must handle the initialization yourself. i cannot emphasize enough the need to carefully check the embedding layer initialization when dealing with any modifications to vocabulary.

as for resources, i'd recommend looking at the "attention is all you need" paper by vaswani et al., it gives great insights about how embeddings are used in transformers, even though its not specifically about embedding initialization, it helps to understand the fundamentals. there is also the book "natural language processing with transformers" by lewis tunstall which contains a very good explanation and practical examples about this problem. finally, look for the documentation for your specific framework, they usually have very complete information about how the embedding layer works, and how to use `masking` of padded tokens.

and, speaking of padding, i once had a colleague who insisted on using spaces for padding, his code was a bit... *spacey* ... i had to actually go to his office and *debug* it. but seriously, check your code carefully and debug every step of the process.

hope this helps and good luck with your model.
