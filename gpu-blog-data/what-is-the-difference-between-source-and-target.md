---
title: "What is the difference between source and target in a PyTorch Transformer?"
date: "2025-01-30"
id: "what-is-the-difference-between-source-and-target"
---
The fundamental distinction between the source and target in a PyTorch Transformer lies in their roles within the sequence-to-sequence learning paradigm.  The source represents the input sequence, while the target represents the desired output sequence.  This distinction is crucial because it dictates the flow of information within the Transformer architecture and influences how attention mechanisms are applied.  My experience optimizing large-scale multilingual translation models solidified this understanding.  Misunderstanding this core concept led to significant performance bottlenecks early in my work.

**1. Clear Explanation**

The Transformer, at its core, processes sequential data.  In many applications, this data is textual, but the architecture generalizes to other sequential modalities.  The "source" sequence provides the initial context for the model. It's the raw input that the Transformer encodes to a latent representation.  This encoding is not a simple mapping but a contextualized representation where each element's meaning is informed by its relationship to other elements in the sequence. The encoder component of the Transformer handles this process.  The output of the encoder is a set of contextualized embeddings, one for each element in the source sequence.

The "target" sequence, on the other hand, is the desired output.  The decoder component of the Transformer uses the encoded representation of the source sequence, along with the previously generated target tokens, to predict the next target token.  This is a step-by-step process; the model generates the target sequence autoregressively, one token at a time.  The decoder leverages self-attention to attend to the previously generated tokens and cross-attention to attend to the encoded source sequence. This allows the model to generate coherent and contextually relevant output.

Crucially, the relationship between source and target is not necessarily symmetric.  While some tasks might have a direct one-to-one correspondence (e.g., machine translation where the source is in one language and the target is in another), other tasks can have significantly different source and target structures. For instance, in text summarization, the source is a lengthy document, and the target is a concise summary. In these asymmetric cases, the Transformer's ability to capture complex relationships between sequences becomes particularly vital.  During my work on a question-answering system, this asymmetry required careful consideration of attention masking to prevent information leakage.

**2. Code Examples with Commentary**

The following examples illustrate the source and target handling in different PyTorch Transformer-based tasks.  These are simplified examples to highlight the core concepts; actual implementations would require considerably more code for data loading, model architecture specification, and training loop management.

**Example 1: Machine Translation (English to French)**

```python
import torch
import torch.nn as nn

# ... (Define encoder and decoder models) ...

source_sentence = ["hello", "world"]
target_sentence = ["bonjour", "le", "monde"]

# Tokenize the sentences (replace with actual tokenization)
source_tokens = torch.tensor([[1, 2]]) # 1: hello, 2: world
target_tokens = torch.tensor([[3, 4, 5]]) # 3: bonjour, 4: le, 5: monde

# Encode the source sentence
source_embeddings = encoder(source_tokens)

# Decode the target sentence step-by-step (teacher forcing for simplicity)
decoder_input = torch.tensor([[3]]) # Start with the <BOS> token
output = []
for i in range(len(target_tokens[0])):
  decoder_output = decoder(decoder_input, source_embeddings)
  predicted_token = torch.argmax(decoder_output, dim=-1)
  output.append(predicted_token)
  decoder_input = torch.cat([decoder_input, predicted_token], dim=1)

print(output) # Predicted target tokens
```

This example demonstrates the basic flow. The encoder processes the source sentence, creating contextualized embeddings.  The decoder then iteratively predicts target tokens, conditioning its predictions on the source embeddings and the previously generated target tokens. Teacher forcing, where the ground truth target token is used at each step, simplifies training but isn't used during inference.

**Example 2: Text Summarization**

```python
import torch
import torch.nn as nn

# ... (Define encoder and decoder models) ...

source_document = "This is a long document that needs to be summarized.  It contains many sentences and paragraphs."
target_summary = "This document requires summarization."

# Tokenize the document and summary
source_tokens = torch.tensor([[1, 2, 3, ...]]) # Tokenized document
target_tokens = torch.tensor([[4, 5, 6]]) # Tokenized summary

# Encoding the source document and decoding the target summary follows a similar pattern to Example 1
#  However, the source and target lengths can significantly differ.
# ... (Encoder and decoder operations as before) ...

```

This example highlights the asymmetry. The source is a long sequence, and the target is considerably shorter.  The decoder learns to compress the information from the source into a concise summary.  The handling of variable sequence lengths requires padding and masking to ensure efficient processing.

**Example 3: Sequence Classification**

```python
import torch
import torch.nn as nn

# ... (Define a Transformer encoder-only model) ...

input_sequence = ["This", "is", "a", "positive", "sentence"]
# Target is the class label (e.g., 1 for positive, 0 for negative)
target_label = torch.tensor([1])

# Tokenize the input sequence
input_tokens = torch.tensor([[1, 2, 3, 4, 5]])

# Encode the input sequence
encoded_representation = encoder(input_tokens)

# Apply a classification layer to the final encoder output
classification_layer = nn.Linear(encoder.output_dim, 2) # 2 classes
logits = classification_layer(encoded_representation[:, -1, :]) # Use last token's representation

# Compute the loss
loss = nn.CrossEntropyLoss()(logits, target_label)

```

This example uses a Transformer encoder only. There's still a source (the input sequence) and a target (the classification label), but the target is not a sequence.  This demonstrates that the source/target distinction applies broadly, not solely to sequence-to-sequence tasks.  The output of the encoder is reduced to a classification score via a linear layer, representing the predicted class.

**3. Resource Recommendations**

"Attention is All You Need" (the original Transformer paper);  "Deep Learning with PyTorch" (a comprehensive text on deep learning using PyTorch);  a reputable textbook on natural language processing.  Thorough understanding of linear algebra and probability theory is also highly beneficial.  Furthermore, studying the source code of established PyTorch Transformer implementations provides invaluable practical insights.  Careful attention to the documentation of PyTorch's `nn.Transformer` module is essential.
