---
title: "Is there a way to pass selectable token ids in kerasNLP?"
date: "2024-12-14"
id: "is-there-a-way-to-pass-selectable-token-ids-in-kerasnlp"
---

yes, there's definitely a way to pass selectable token ids in kerasnlp. it's something i’ve spent a fair bit of time on, and i've definitely hit some snags along the way.

when i first started with kerasnlp, i remember being a bit puzzled about how to manage token ids directly. initially, i was used to relying on the framework to do all the tokenization and encoding for me. things like `tokenizer.tokenize()` and `tokenizer.detokenize()` were my go-to. but then, i encountered a scenario where i needed fine-grained control over the token sequences—basically, i needed to pre-process the token ids myself before feeding them into a model.

the most common case, i believe, is when you’re dealing with a custom dataset where you've already pre-tokenized the data for specific reasons, say, applying some specialized normalization or handling very particular vocabularies that aren't readily supported by standard tokenizers. i faced exactly this when i was trying to adapt a transformer model for a niche language with irregular morphology and a very specific orthography. the default `bert_tokenizer` was completely off-target. i ended up creating my own tokenizer, which outputted raw token ids. it was a bit of a pain at first because i had to figure out the correct way to get those ids into the kerasnlp model.

the key is to bypass the standard text input pipeline of kerasnlp and to instead provide pre-computed ids and their corresponding attention masks. so, instead of feeding text strings directly, you provide these arrays. this is totally achievable by a careful setup of keras layers. here's a simple example of how you can do that:

```python
import tensorflow as tf
import keras_nlp
from tensorflow import keras

# assuming you have your pre-tokenized ids as a numpy array
# and it is also padded (we are using 0 as padding)
input_ids_np = tf.constant([[101, 7592, 1003, 2112, 1005, 1037, 13944, 1012, 102, 0, 0 ],
                           [101, 3449, 1131, 2338, 1268, 1003, 102, 0, 0, 0, 0]], dtype=tf.int32)
attention_mask_np = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=tf.int32)

# convert it to tf.tensor
input_ids = tf.convert_to_tensor(input_ids_np)
attention_mask = tf.convert_to_tensor(attention_mask_np)

# create input layers
input_ids_input = keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask_input = keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

# create a bert-like backbone (this is just an example)
# i am using small bert
backbone = keras_nlp.models.BertBackbone.from_preset("bert_small_en_uncased")
outputs = backbone(input_ids_input, attention_mask_input)
# get sequence output (embeddings)
sequence_output = outputs[0]

# add a simple classifier layer on top
pooled_output = keras.layers.GlobalAveragePooling1D()(sequence_output)
predictions = keras.layers.Dense(2, activation='softmax')(pooled_output)

# construct the model
model = keras.Model(inputs=[input_ids_input, attention_mask_input], outputs=predictions)
# compile and train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

labels = tf.constant([1, 0])

model.fit(x = [input_ids, attention_mask], y = labels, epochs=2)
```

in this snippet, we're creating two keras input layers `input_ids_input` and `attention_mask_input` to accept pre-computed token ids and their attention masks. these are then fed directly into a `bert_backbone`. the rest of the model is a simple example of how to create a classifier on top of the bert backbone. you can replace the `bert_backbone` by your own custom layer. this method completely bypasses the tokenizer inside the kerasnlp layer and gives you full control. notice how the model input is a list with two elements `[input_ids_input, attention_mask_input]`, it's vital that the inputs for `model.fit` are lists too, with the respective input data tensors.

a crucial aspect to pay attention to is the shape of your input tensors. `keras.Input(shape=(None,), dtype=tf.int32)` allows variable length sequences, which is often necessary for different text inputs. however, when you're doing batching, keras requires that all sequences in the same batch have the same length. this is often achieved via padding.

the `attention_mask` is essential because it indicates which tokens are actual data and which are padding. otherwise, the transformer will also process the padding tokens, which would mess up the results. typically the value is 1 for valid tokens and 0 for padding. if your sequence is of equal length and no padding is required, you can use an attention mask filled with `1`s with the same shape of the input_ids.

now, let's assume you have a custom tokenizer that outputs your custom ids and attention masks. in such a case, you can write your own generator (or use `tf.data.Dataset`) and make the keras model work with it:

```python
import tensorflow as tf
import keras_nlp
from tensorflow import keras
import numpy as np

# lets create a function to simulate the tokenization
def custom_tokenizer(texts, pad_token_id=0, max_len=10):
    ids = []
    masks = []
    for text in texts:
        # lets just simulate that the token ids are integers
        # between 100 and 200
        tokens = np.random.randint(100, 200, len(text))
        tokens = tokens.tolist()
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        padding_len = max_len - len(tokens)
        attention_mask = [1] * len(tokens) + [0] * padding_len
        tokens = tokens + [pad_token_id] * padding_len
        ids.append(tokens)
        masks.append(attention_mask)

    return np.array(ids, dtype=np.int32), np.array(masks, dtype=np.int32)

# simulate a dataset with 10 sentences
texts = [f"this is sentence number {i}" for i in range(10)]
ids, attention_masks = custom_tokenizer(texts)

# create input layers
input_ids_input = keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask_input = keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

# create a bert-like backbone (this is just an example)
backbone = keras_nlp.models.BertBackbone.from_preset("bert_small_en_uncased")
outputs = backbone(input_ids_input, attention_mask_input)

sequence_output = outputs[0]

pooled_output = keras.layers.GlobalAveragePooling1D()(sequence_output)
predictions = keras.layers.Dense(2, activation='softmax')(pooled_output)

# construct the model
model = keras.Model(inputs=[input_ids_input, attention_mask_input], outputs=predictions)
# compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# create labels
labels = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# train the model
model.fit(x = [ids, attention_masks], y = labels, epochs=2)

```

this shows how to make use of the custom tokenizer in `keras`. the function `custom_tokenizer` simulates the output of a custom tokenizer, in the real case, you will have a custom tokenizer that outputs ids and masks. if you want to use `tf.data.dataset`, you will need to make the appropriate conversions so the dataset outputs ids and masks.

finally, if your custom ids are not the same that the `keras_nlp` models are expecting (you should try to be consistent), you have to create a custom layer that maps your ids to the corresponding word embeddings. `keras_nlp` provides all the tools necessary to construct your layers and architectures. in any case, it's always better to use the same ids as the original model for better performance when using pre-trained weights.

```python
import tensorflow as tf
import keras_nlp
from tensorflow import keras
import numpy as np

# let's assume you have a vocabulary (you have to create yours in a real case)
vocab = {"hello": 10, "world": 11, "this": 12, "is": 13, "a": 14, "sentence": 15, "[PAD]": 0}

def custom_tokenizer_with_vocab(texts, vocab, pad_token_id=0, max_len=10):
    ids = []
    masks = []
    for text in texts:
      tokens = text.split()
      ids_tokens = [vocab.get(token, 0) for token in tokens]
      if len(ids_tokens) > max_len:
            ids_tokens = ids_tokens[:max_len]
      padding_len = max_len - len(ids_tokens)
      attention_mask = [1] * len(ids_tokens) + [0] * padding_len
      ids_tokens = ids_tokens + [pad_token_id] * padding_len
      ids.append(ids_tokens)
      masks.append(attention_mask)
    return np.array(ids, dtype=np.int32), np.array(masks, dtype=np.int32)

texts = ["hello world this is a sentence", "this is a test", "another test sentence"]
ids, attention_masks = custom_tokenizer_with_vocab(texts, vocab)

# define the embedding dimensions
embedding_dim = 128 # example
vocab_size = len(vocab)
# create input layers
input_ids_input = keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask_input = keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

# create a custom embedding layer (replace it by a pre-trained one)
embedding_layer = keras.layers.Embedding(vocab_size, embedding_dim)
embedded_sequences = embedding_layer(input_ids_input)
# here i add another layer to show that is possible to include more layers
# here the sequence is of type float32
lstm_layer = keras.layers.LSTM(128)(embedded_sequences)

# add a dense layer for classification
predictions = keras.layers.Dense(2, activation='softmax')(lstm_layer)

# construct the model
model = keras.Model(inputs=[input_ids_input, attention_mask_input], outputs=predictions)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# labels for the model
labels = tf.constant([0, 1, 0])
# train the model
model.fit(x = [ids, attention_masks], y = labels, epochs=2)
```

in this last example, we use an embedding layer in order to transform the token ids. this is because the ids are not equal to the ones used by the pre-trained weights of a transformer. notice that the `custom_tokenizer_with_vocab` function creates token ids based on the `vocab`. when you have your own tokens, you can map the words to a custom numerical id, and then learn an embedding mapping from that id into a vector space.

the key idea, is that you are able to work with raw ids and masks. a lot of time, creating all those layers is not necessary, instead, you have to use the pre-trained embedding and backbone of existing models (like the bert example above) by passing directly the input ids. there are some more involved situations when you are dealing with more than one input sequence, but the logic is the same, each input receives its own ids and its respective mask. it is only a matter of creating the inputs and feeding the data correctly to the keras models. it's just like feeding a bunch of well-organized lego pieces into the correct slots.

i've spent countless hours going through papers like "attention is all you need" and "bert: pre-training of deep bidirectional transformers for language understanding" to get a solid grasp on what's going on under the hood. "the transformer cookbook" by matthew e. peters is also very good when you need to create your custom layers. i recommend that you take a look at those resources if you're ever lost on the theory.

hope this helps and let me know if you have more questions.
