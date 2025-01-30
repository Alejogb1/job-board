---
title: "How do I access the tokenizer after loading a custom BERT model saved with Keras and TF2?"
date: "2025-01-30"
id: "how-do-i-access-the-tokenizer-after-loading"
---
The core issue when working with custom BERT models in TensorFlow 2 and Keras, specifically after saving and reloading them, lies in the separation of model weights and the associated preprocessing logic. The tokenizer, responsible for converting text into numerical inputs, is not intrinsically embedded within the model itself. This crucial piece often requires explicit handling upon model loading. Through my work optimizing various natural language pipelines, this has become a frequent consideration and is not unique to BERT but applies broadly to models relying on pre-processing.

When loading a BERT model, you are typically re-instantiating the architecture and its associated weights. The saved model file, whether it's a SavedModel format or HDF5, contains the learned parameters. It does *not* inherently store the preprocessing artifacts—like the wordpiece vocabulary or the specific tokenization parameters—used during training. These must be preserved and accessed separately. Therefore, to tokenize new input data correctly for your reloaded model, you must explicitly reload the tokenizer configuration alongside the model.

The process revolves around two primary steps: first, saving both the model and the tokenizer; and secondly, loading them separately and appropriately. The common oversight involves saving only the model and assuming the tokenizer is somehow inferred or included. This is rarely the case, particularly with custom models. The tokenizer is itself an independent object within TensorFlow Text or associated libraries like Hugging Face Transformers.

Here's how this manifests in practice, supported by three relevant code examples.

**Example 1: Saving and Loading Model and Tokenizer (Correct Approach)**

This first example demonstrates the correct approach by explicitly saving and subsequently loading the tokenizer along with the model. I’m using `keras.saving.save_model` and `keras.saving.load_model`, as they're standard for a modern Keras workflow.

```python
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras

# Assume we have a pre-trained tokenizer from a library
# Here for demonstration, I'll just build a dummy tokenizer
vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'hello', 'world']
tokenizer = tf_text.WordpieceTokenizer(vocab=vocab, unknown_token='[UNK]')

# Assume we have a pre-trained BERT model from Keras
# For brevity, I'll build a simple dummy model
input_ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
embedding = keras.layers.Embedding(input_dim=len(vocab), output_dim=16)(input_ids)
output = keras.layers.GlobalAveragePooling1D()(embedding)
output = keras.layers.Dense(1, activation="sigmoid")(output)
model = keras.Model(inputs=input_ids, outputs=output)

# 1. Save the model
model.save("my_bert_model")

# 2. Save the tokenizer configuration
import json
tokenizer_config = {
    "vocab": vocab,
    "unknown_token": '[UNK]'
}

with open("my_tokenizer.json", "w") as f:
    json.dump(tokenizer_config, f)


# Load the model from file.
loaded_model = keras.saving.load_model("my_bert_model")


# Load the tokenizer from config
with open("my_tokenizer.json", "r") as f:
    tokenizer_config = json.load(f)
loaded_tokenizer = tf_text.WordpieceTokenizer(vocab=tokenizer_config["vocab"], unknown_token=tokenizer_config["unknown_token"])


# Using the loaded tokenizer to prepare the input data

text_input = tf.constant(["hello world", "hello hello"])
tokens = loaded_tokenizer.tokenize(text_input)
input_ids = tokens.to_tensor(default_value=0)

prediction = loaded_model(input_ids)
print("Prediction:", prediction)
```

In this example, I first save the trained model to a directory named `my_bert_model`, standard practice with Keras. Secondly, I extract tokenizer parameters and serialize them into a JSON file (`my_tokenizer.json`). After reloading both entities, I use the tokenizer to convert text inputs into tensors digestible by the model. It is critical to understand here that saving and loading the model as a whole unit (like using `model.save()` ) by itself does not capture the pre-processing rules.

**Example 2: Incorrect Approach: Loading Only the Model**

This example illustrates the error and resulting issues when loading only the model without the tokenizer. Many beginner's face this scenario:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np


# Assume same model as Example 1

vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'hello', 'world']
input_ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
embedding = keras.layers.Embedding(input_dim=len(vocab), output_dim=16)(input_ids)
output = keras.layers.GlobalAveragePooling1D()(embedding)
output = keras.layers.Dense(1, activation="sigmoid")(output)
model = keras.Model(inputs=input_ids, outputs=output)

# 1. Save the model
model.save("my_bert_model_incorrect")

# Assume the user does not save the tokenizer information.

# 2. Load the model
loaded_model = keras.saving.load_model("my_bert_model_incorrect")


# Attempting to use the model without the correct tokenizer
# Providing arbitrary input
arbitrary_input = np.array([[1, 2, 3]]) # These are *not* the correct input ids generated from the original tokenizer


prediction = loaded_model(arbitrary_input)

print("Prediction with arbitrary input:", prediction)

# Attempting to directly predict text will raise an error since the model expects the encoded inputs
try:
  text_input = tf.constant(["hello world", "hello hello"])
  prediction_text = loaded_model(text_input)
  print("Prediction:", prediction_text)

except Exception as e:
  print(e)

```

Here, the model is saved, then reloaded, but without any associated tokenizer. The user then attempts to pass arbitrary integer inputs, and while it won't crash, results will be meaningless because the input does not adhere to the model's expected input domain. Furthermore, attempting to pass textual data directly to the model results in a `tf.errors.InvalidArgumentError`, highlighting the error because the model expects tensor inputs, not raw strings. The error stems from the lack of a transformation process (tokenization) that prepares the text for consumption by the model.

**Example 3: Loading with a Different Tokenizer Configuration (Problematic)**

This final example demonstrates a common mistake: attempting to load and use the model with a different tokenizer than the one it was originally trained on. This happens more often than is recognized and stems from not meticulously managing training pipelines and inference pipelines together.

```python
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras
import numpy as np
import json

# Original model and tokenizer (same as in Example 1)
vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'hello', 'world']
tokenizer = tf_text.WordpieceTokenizer(vocab=vocab, unknown_token='[UNK]')
input_ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
embedding = keras.layers.Embedding(input_dim=len(vocab), output_dim=16)(input_ids)
output = keras.layers.GlobalAveragePooling1D()(embedding)
output = keras.layers.Dense(1, activation="sigmoid")(output)
model = keras.Model(inputs=input_ids, outputs=output)


model.save("my_bert_model_config_mismatch")
tokenizer_config = {
    "vocab": vocab,
    "unknown_token": '[UNK]'
}
with open("my_tokenizer.json", "w") as f:
    json.dump(tokenizer_config, f)


# Assume we accidentally load a *different* tokenizer config
new_vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'apple', 'banana']
new_tokenizer_config = {
    "vocab": new_vocab,
    "unknown_token": '[UNK]'
}

with open("my_new_tokenizer.json", "w") as f:
    json.dump(new_tokenizer_config,f)

# Load the model (same as before).
loaded_model = keras.saving.load_model("my_bert_model_config_mismatch")

# Load the *different* tokenizer from a config.
with open("my_new_tokenizer.json", "r") as f:
  tokenizer_config = json.load(f)
loaded_tokenizer = tf_text.WordpieceTokenizer(vocab=tokenizer_config["vocab"], unknown_token=tokenizer_config["unknown_token"])

# Tokenize input data using the *incorrect* tokenizer
text_input = tf.constant(["hello world"])
tokens = loaded_tokenizer.tokenize(text_input)
input_ids = tokens.to_tensor(default_value=0)


# Attempt to make predictions
prediction = loaded_model(input_ids)

print("Prediction with mismatched tokenizer:", prediction)
```

The saved model has a specific vocabulary embedded in the embedding layer. However, the code attempts to load a *different* tokenizer, which tokenizes the input text using a different vocabulary and token mapping. Because the input IDs to the model now reflect a different vocabulary, the subsequent predictions will be nonsensical. This illustrates why matching tokenizers during inference with the training tokenizers is critical. The model is essentially receiving completely different numerical inputs compared to those during training.

**Resource Recommendations**

For a thorough understanding, I recommend exploring the TensorFlow official documentation. Specifically, review the sections on saving and loading Keras models (`tf.keras.saving` or `keras.saving`) and the corresponding text processing modules in `tensorflow_text` . There are also numerous guides about using transformers within TensorFlow and Keras ecosystems. Also, delving into the documentation specific to the Hugging Face Transformers library (if that's your particular stack) for tokenization best practices is advisable. While I’ve avoided using actual links due to constraints, focusing on documentation with keywords such as "keras saving models," "tensorflow text tokenization," "wordpiece tokenizer," and "transformers tokenizers" within these resources should guide you towards a robust solution. Furthermore, explore tutorials dealing with custom datasets and models that use transformers. Pay special attention to the part where the tokenizer is initialized, saved, and reloaded. These steps are where a lot of problems hide.
