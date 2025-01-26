---
title: "How can I manually insert 'SEP' tokens using a TensorFlow Hub BERT preprocessor?"
date: "2025-01-26"
id: "how-can-i-manually-insert-sep-tokens-using-a-tensorflow-hub-bert-preprocessor"
---

The TensorFlow Hub BERT preprocessor does not inherently expose direct manipulation of token IDs; instead, it manages the tokenization, ID mapping, and tensor construction internally. My experience working on NLP pipelines for semantic search revealed the need to fine-tune sequence handling, including manual insertion of the [SEP] token, primarily when dealing with complex, multi-part text inputs not natively supported by standard BERT inputs.

The default BERT preprocessor expects either a single text string or a pair of strings. If you're feeding in data already pre-segmented, perhaps from a document structure, or trying to concatenate different input segments with specific separations, the standard processing will not achieve the precise token sequence you desire. Specifically, it might inappropriately split segments or lack explicit [SEP] token insertion where needed between manually constructed sequence segments. Direct manipulation of the token IDs is necessary because relying solely on preprocessor's string inputs limits control over the final token sequence.

The core challenge lies in the design of BERT's input requirements and the preprocessor’s handling. BERT requires a sequence of token IDs corresponding to the processed input. These token IDs are mapped from string tokens using a vocabulary lookup. The default preprocessor uses this vocab and includes functions for padding and truncation but hides the actual ID sequence composition. To manually include the [SEP] token, therefore, it is required to perform the tokenization step ourselves using the preprocessor’s vocabulary object, assemble a desired sequence of IDs incorporating the [SEP] tokens and other token segments as required, and then create a tensor representation. This approach bypasses some of the preprocessor’s automatic string-handling logic.

The key is to extract the BERT preprocessor's tokenizer, which has the required vocabulary mapping, and to use its `tokenize()` method. We can then retrieve the `token_ids` from the tokenized outputs. Subsequently, the token IDs are manipulated and combined and converted into a suitable tensor format for feeding into the BERT model.

Let’s see examples demonstrating this manual token insertion strategy, beginning with a straightforward two-sentence sequence.

```python
import tensorflow as tf
import tensorflow_hub as hub

# Load a BERT preprocessor from TF Hub
preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
preprocessor = hub.KerasLayer(preprocessor_url)
# Extract the tokenizer from the preprocessor
tokenizer = preprocessor.tokenizer

# Input sentences as a list
sentence1 = "This is the first sentence."
sentence2 = "This is the second sentence."

# Manually tokenize the input sentences
tokens1 = tokenizer.tokenize([sentence1])
tokens2 = tokenizer.tokenize([sentence2])

# Convert tokens to IDs
ids1 = tokenizer.lookup(tokens1)
ids2 = tokenizer.lookup(tokens2)

# Get the ID for the [SEP] token
sep_id = tokenizer.lookup(['[SEP]'])[0]
cls_id = tokenizer.lookup(['[CLS]'])[0]

# Build the manual token ID sequence
manual_ids = tf.concat([
    tf.constant([cls_id]),
    ids1[0],
    tf.constant([sep_id]),
    ids2[0],
    tf.constant([sep_id])
    ], axis=0)


# Pad or truncate the sequence to BERT’s expected length
max_seq_length = 128  # Example maximum sequence length
padding_value = 0

if tf.shape(manual_ids)[0] > max_seq_length:
  manual_ids = manual_ids[:max_seq_length] # Truncate if necessary
else:
  padding_len = max_seq_length - tf.shape(manual_ids)[0]
  manual_ids = tf.concat([manual_ids, tf.zeros([padding_len], dtype=tf.int64)], axis=0) # Pad if required

# Convert to tensor
input_ids = tf.expand_dims(manual_ids, axis=0)


# Create the attention mask (1 for real tokens, 0 for padding)
attention_mask = tf.where(input_ids != 0, 1, 0)

print("Input IDs:", input_ids)
print("Attention mask:", attention_mask)
```

In the above example, after extracting the tokenizer, I manually tokenized `sentence1` and `sentence2` and retrieved the corresponding IDs. The `[SEP]` token ID was then retrieved separately. These IDs were concatenated together to form the full input sequence, also including the `[CLS]` token at the start. The sequence was then padded to fit a predefined maximum sequence length. Finally, the input IDs and an attention mask tensor were created, ready to feed into BERT.

Now, let’s look at a slightly more complex case, including multiple segments requiring distinct separator tokens. This example might represent input from multiple document sections.

```python
import tensorflow as tf
import tensorflow_hub as hub

# Load a BERT preprocessor from TF Hub
preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
preprocessor = hub.KerasLayer(preprocessor_url)
tokenizer = preprocessor.tokenizer

# Multiple text segments
segment1 = "First section of the document."
segment2 = "A second segment here."
segment3 = "The final paragraph."

# Tokenize each segment and get IDs
tokens1 = tokenizer.tokenize([segment1])
tokens2 = tokenizer.tokenize([segment2])
tokens3 = tokenizer.tokenize([segment3])

ids1 = tokenizer.lookup(tokens1)
ids2 = tokenizer.lookup(tokens2)
ids3 = tokenizer.lookup(tokens3)

# Get [SEP] and [CLS] IDs
sep_id = tokenizer.lookup(['[SEP]'])[0]
cls_id = tokenizer.lookup(['[CLS]'])[0]


# Construct manual token sequence with separators
manual_ids = tf.concat([
    tf.constant([cls_id]),
    ids1[0],
    tf.constant([sep_id]),
    ids2[0],
    tf.constant([sep_id]),
    ids3[0],
    tf.constant([sep_id])
    ], axis=0)


# Pad/truncate to a maximum sequence length
max_seq_length = 128
padding_value = 0

if tf.shape(manual_ids)[0] > max_seq_length:
    manual_ids = manual_ids[:max_seq_length]
else:
    padding_len = max_seq_length - tf.shape(manual_ids)[0]
    manual_ids = tf.concat([manual_ids, tf.zeros([padding_len], dtype=tf.int64)], axis=0)


# Create input IDs tensor
input_ids = tf.expand_dims(manual_ids, axis=0)

# Create attention mask
attention_mask = tf.where(input_ids != 0, 1, 0)


print("Input IDs:", input_ids)
print("Attention mask:", attention_mask)
```

Here, the same principles are applied, but now for three text segments. Each segment is tokenized and the ids retrieved and are combined manually by including separator tokens. The sequence is then padded and attention mask computed to finalize the format required by BERT.

The final example focuses on segment type IDs. While not strictly about inserting [SEP] tokens, the need for fine control often extends to managing segment types. Here, I am showing how to manually add a segment type id tensor which informs the model about which segment of the input each token belongs to.

```python
import tensorflow as tf
import tensorflow_hub as hub

# Load a BERT preprocessor from TF Hub
preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
preprocessor = hub.KerasLayer(preprocessor_url)
tokenizer = preprocessor.tokenizer

# Input text segments
segment1 = "This is the first segment."
segment2 = "This is the second segment."

# Tokenize segments and get IDs
tokens1 = tokenizer.tokenize([segment1])
tokens2 = tokenizer.tokenize([segment2])

ids1 = tokenizer.lookup(tokens1)
ids2 = tokenizer.lookup(tokens2)

# Get [SEP] and [CLS] IDs
sep_id = tokenizer.lookup(['[SEP]'])[0]
cls_id = tokenizer.lookup(['[CLS]'])[0]


# Manual token sequence creation
manual_ids = tf.concat([
    tf.constant([cls_id]),
    ids1[0],
    tf.constant([sep_id]),
    ids2[0],
    tf.constant([sep_id])
    ], axis=0)


# Create segment type IDs (0 for segment 1, 1 for segment 2)
segment_ids = tf.concat([
    tf.zeros(tf.shape(tf.constant([cls_id]))),
    tf.zeros(tf.shape(ids1[0])),
    tf.zeros(tf.shape(tf.constant([sep_id]))),
    tf.ones(tf.shape(ids2[0])),
    tf.zeros(tf.shape(tf.constant([sep_id])))
], axis=0)

# Ensure both sequence have the same length
max_seq_length = 128
padding_value = 0


if tf.shape(manual_ids)[0] > max_seq_length:
    manual_ids = manual_ids[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
else:
    padding_len = max_seq_length - tf.shape(manual_ids)[0]
    manual_ids = tf.concat([manual_ids, tf.zeros([padding_len], dtype=tf.int64)], axis=0)
    segment_ids = tf.concat([segment_ids, tf.zeros([padding_len], dtype=tf.int64)], axis=0)

# Input IDs tensor and Segment ID tensor
input_ids = tf.expand_dims(manual_ids, axis=0)
segment_ids = tf.expand_dims(segment_ids, axis=0)


# Create attention mask
attention_mask = tf.where(input_ids != 0, 1, 0)


print("Input IDs:", input_ids)
print("Segment IDs:", segment_ids)
print("Attention mask:", attention_mask)

```
In this last example, in addition to the manual creation of the input ID sequence, we also generated a segment ID tensor, with 0 corresponding to the first and 1 corresponding to the second segment. This can be extremely important for inputs where different segments have very different semantics, as is the case with complex question answering or text summarization tasks. This highlights the level of manual control that can be achieved when directly manipulating token IDs.

For further exploration of BERT and similar models, resources from the Hugging Face library offer both conceptual and practical examples. Works by the authors of the original BERT paper, as well as the TensorFlow official documentation, provide solid foundations. Also, the specific documentation for TensorFlow Hub models gives additional specific insights into the design of the preprocessors. I have found exploring NLP courses on platforms like Coursera, Udemy, and fast.ai to offer a practical view of these methods.
